import numpy as np
import progressbar
import random
import gym
from collections import deque
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam


class LearningAgent:
    def __init__(self, env, optimizer):
        ## Initializing the attributes
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.optimizer = optimizer
        ## A special list that provides a O(1) complexity
        ## for append and pop operations as compared to
        ## a normal list that provides a O(n) complexity
        ## for those operations
        self.experience_replay = deque(maxlen = 2000)
        ## Initializing the discount and
        ## exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1
        ## Builds the necessary networks
        #In Deep Q-Learning TD-Target y_i and Q(s,a) are estimated
        # separately by two different neural networks, which are often
        # called the Target-, and Q-Networks (Fig. 4). The parameters
        # θ(i-1) (weights, biases) belong to the Target-Network while
        #  θ(i) belong to the Q-Network.
        self.q_network = self.build_compile_model()
        self.target_network = self.build_compile_model()
        self.align_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))

    















if __name__ == '__main__':
    env = gym.make("Taxi-v2").env
    env.render()

    print(f'Number of states: {env.observation_space.n}')
    print(f'Number of actions: {env.action_space.n}')


