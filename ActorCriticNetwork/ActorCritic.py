from typing import Tuple
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

# Build the actual model to run
class ActorCritic(keras.Model):
    def __init__ (self, num_actions, num_hidden_units):
        """
          Builds an actor critic network

          Args:
            num_actions: Number of possible actions the model can take
            num_hidden_units: Number of units in the common hidden layer of the neural network
        """
        super.__init__()
        # Build Layers
        self.common = keras.layers.Dense(num_hidden_units, activation='relu')
        self.actor = keras.layers.Dense(num_actions)
        self.critic = keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
          Runs the actor critic netwrok

          Args:
            inputs: The observations from the environment to make an action choice on and calculate the reward for.
        """
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

# Allow model to step through simulation
def env_step_wrapper(env):
    """
      Creates the function that allows the model to interact with a certain environment.

      Args:
        env: The environment the model will interact with
    """
    def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
          Returns the environments state, reward, and doneness after an action is taken

          Args:
            action: The action performed on the environment
        """
        state, reward, done, _ = env.step(action)
        return state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32)
    return env_step