import tensorflow as tf
from tensorflow.keras import layers

class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, n_ads):
        super(ActorCriticNetwork, self).__init__()
        self.shared = layers.Dense(128, activation="relu")
        self.actor = layers.Dense(n_ads, activation="softmax")  # Policy output
        self.critic = layers.Dense(1)  # Value function output

    def call(self, state):
        x = self.shared(state)
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

# Initialize the network and load weights
n_ads = 10  # Number of ads
network = ActorCriticNetwork(n_ads)
network.build(input_shape=(None, n_ads))  # Define input shape
network.load_weights("actor_critic.weights.h5")  # Load saved weights