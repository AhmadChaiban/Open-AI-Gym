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

    ## Building the actual Neural Network
    def build_compile_model(self):
        model = Sequential([
            Embedding(self.state_size, 10, input_length = 1),
            Reshape((10,)),
            Dense(50, activation ='relu'),
            Dense(50, activation = 'relu'),
            Dense(self.action_size, activation = 'linear')
        ])

        ## Compiling the model with mean squared error and optimizer
        model.compile(loss = 'mse', optimizer = self.optimizer)
        return model
    ## This gets the wieghts from the Q network model and gives them to the
    ## DQ network
    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    ## Sample an action from the environment using epsilon greedy
    ## or try to predict on the state and return the index of the
    ## maximum of the q_values
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return env.action_space.sample()

        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    















if __name__ == '__main__':
    env = gym.make("Taxi-v2").env
    env.render()

    print(f'Number of states: {env.observation_space.n}')
    print(f'Number of actions: {env.action_space.n}')


