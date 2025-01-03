import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

# A2C Network Architecture
class A2CNetwork(Model):
    def __init__(self, state_dim, action_dim):
        super(A2CNetwork, self).__init__()
        
        # Shared layers
        self.hidden1 = Dense(64, activation='relu')
        self.hidden2 = Dense(32, activation='relu')
        
        # Actor (policy) layers
        self.actor_hidden = Dense(32, activation='relu')
        self.actor_output = Dense(action_dim, activation='softmax')
        
        # Critic (value) layers
        self.critic_hidden = Dense(32, activation='relu')
        self.critic_output = Dense(1, activation=None)
    
    def call(self, state):
        # Shared layers
        shared = self.hidden1(state)
        shared = self.hidden2(shared)
        
        # Actor path
        actor = self.actor_hidden(shared)
        action_probs = self.actor_output(actor)
        
        # Critic path
        critic = self.critic_hidden(shared)
        value = self.critic_output(critic)
        
        return action_probs, value

class AdRecommender:
    def __init__(self, state_dim=10, action_dim=10, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99  # discount factor
        self.network = A2CNetwork(state_dim, action_dim)
        self.optimizer = Adam(learning_rate)
        
        # Initialize experience buffer
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.values_buffer = []
        
        # Load CTR dataset
        self.ctr_data = self.load_ctr_dataset()
        
    def load_ctr_dataset(self):
        try:
            df = pd.read_csv('Ads_CTR_Optimisation.csv')
            return df.values
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return np.zeros((10000, 10))
    
    def get_state(self, username):
        """Get current state (user preferences) from users.json"""
        with open("users.json", "r") as file:
            users = json.load(file)
            user = next((u for u in users if u["username"] == username), None)
            if user:
                state = [user.get(str(i), 0.0) for i in range(self.action_dim)]
                return np.array(state, dtype=np.float32)
        return np.zeros(self.action_dim, dtype=np.float32)
    
    def recommend(self, username, top_k=10):
        """Get recommendations using the A2C network"""
        state = self.get_state(username)
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        
        # Get action probabilities and value from network
        action_probs, value = self.network(state)
        action_probs = action_probs.numpy()[0]
        
        # Select top-k actions based on probabilities
        recommendations = np.argsort(action_probs)[-top_k:][::-1]
        
        # Store state and value for learning
        self.states_buffer.append(state[0])
        self.values_buffer.append(value[0][0])
        
        return recommendations.tolist()
    
    def update(self, state, action, reward, next_state):
        """Update the A2C network using the experience"""
        with tf.GradientTape() as tape:
            # Current prediction
            action_probs, value = self.network(tf.convert_to_tensor([state], dtype=tf.float32))
            
            # Next state value prediction
            _, next_value = self.network(tf.convert_to_tensor([next_state], dtype=tf.float32))
            
            # Calculate advantage
            advantage = reward + self.gamma * next_value - value
            
            # Actor loss
            action_onehot = tf.one_hot(action, self.action_dim)
            log_prob = tf.math.log(action_probs + 1e-10)
            actor_loss = -tf.reduce_sum(action_onehot * log_prob * advantage)
            
            # Critic loss
            critic_loss = tf.square(advantage)
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss
        
        # Calculate gradients and update network
        grads = tape.gradient(total_loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        
        return total_loss.numpy()

# Flask application setup
app = Flask(__name__)
CORS(app)

# Initialize recommender
recommender = AdRecommender()

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        username = data.get("username")
        
        if not username:
            return jsonify({"error": "Username not provided"}), 400
        
        # Get state and predictions from A2C model
        state = recommender.get_state(username)
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs, _ = recommender.network(state_tensor)
        
        # Convert to numpy array and get probabilities
        probabilities = action_probs.numpy()[0] * 100  # Convert to percentage
        
        # Get top recommendations
        recommendations = np.argsort(probabilities)[-10:][::-1]
        
        # Create response with recommendations and their confidence scores
        response = {
            "recommended_ads": recommendations.tolist(),
            "confidence_scores": {
                str(ad_id): float(probabilities[ad_id])  # Convert to native Python types
                for ad_id in recommendations
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.json
        username = data.get("username")
        liked_ads = data.get("feedback", {}).get("liked_ads", [])
        
        if not username or not liked_ads:
            return jsonify({"error": "Missing username or liked_ads"}), 400
        
        # Get current state
        current_state = recommender.get_state(username)
        
        # Process feedback for each liked ad
        for ad_id in liked_ads:
            # Update user preferences in users.json
            with open("users.json", "r+") as file:
                users = json.load(file)
                user_index = next((i for i, user in enumerate(users) 
                                 if user["username"] == username), None)
                if user_index is not None:
                    # Update preference
                    users[user_index][str(ad_id)] = min(
                        users[user_index].get(str(ad_id), 0) + 0.1,
                        1.0
                    )
                    # Save updated preferences
                    file.seek(0)
                    json.dump(users, file, indent=4)
                    file.truncate()
            
            # Get new state after update
            next_state = recommender.get_state(username)
            
            # Calculate reward based on historical CTR data
            reward = 1.0 if ad_id in liked_ads else 0.0
            
            # Update A2C network
            recommender.update(current_state, ad_id, reward, next_state)
        
        return jsonify({"message": "Feedback processed successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001)