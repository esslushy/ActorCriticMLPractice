from typing import List, Tuple
import gym
import tensorflow as tf
import numpy as np
import argparse

from tensorflow.python.keras.backend import dtype
from ActorCritic import ActorCritic, env_step_wrapper

parser = argparse.ArgumentParser('Choose way to run the Cart Pole')
parser.add_argument('--random', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--model', action='store_true')
args = parser.parse_args()

# Make environment model will train in
env = gym.make('CartPole-v1')

# Set random seed
seed = 7
tf.random.set_seed(seed)
np.random.seed(seed)
env.seed(seed)

# Run random actions in environment
if args.random:
    for _ in range(20):
        observation = env.reset()
        for _ in range(250):
            # Render the current state of the environment
            env.render()
            # Report what the computer can see
            print(observation)
            # Create a random action
            action = env.action_space.sample() 
            # Perform that action
            observation, reward, done, info = env.step(action)
            if done:
                break
    env.close()
# Train the model to balance the stick
elif args.train:
    # Create the model with the number of outputs equal to the number of actions and 128 nodes in the hidden layer
    model = ActorCritic(env.action_space.n, 128)
    # Create TensorFlow function that allows model to interact with step on environment
    def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
        tf.numpy_function(env_step_wrapper(env), [action], [tf.float32, tf.int32, tf.int32])
    # Create function that gathers information for a full episode of CartPole (from start to when done=True)
    def run_episode(initial_state: tf.Tensor, model: tf.keras.Model, max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Create arrays to gather information
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # Gather initial information
        initial_state_shape = initial_state.shape()
        state = initial_state # redefined for for loop


# Watch the network balance the stick
elif args.model:
    pass
# If no choices were passed to choose
else:
    print('Please choose an option to run the system') 

env.close()