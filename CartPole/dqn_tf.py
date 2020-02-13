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

    ## In this method we pick random samples from the experience
    ## replay memory and train the Q-Network
    def retrain(self, batch_size):
        ## Taking a minibatch
        minibatch = random.sample(self.experience_replay, batch_size)
        ## Finding the state, action, reward and state prime in the batch
        for state, action, reward, next_state, terminated in minibatch:
            ## Making a prediction on the state
            target = self.q_network.predict(state)
            ## If it's terminated just return the reward
            if terminated:
                target[0][action] = reward
            ## Else predict on the next state
            ## and (NOT SURE OF THIS ONE) discount the reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            ## Fit the network on the state, target for n epochs
            self.q_network.fit(state, target, epochs=1, verbose=0)

    def play_one_episode(self, env, state, bar, agent, timesteps_per_episode, batch_size):
        for timestep in range(timesteps_per_episode):
            # Run Action
            action = agent.act(state)
            # Take action
            next_state, reward, terminated, info = env.step(action)
            next_state = np.reshape(next_state, [1, 1])
            agent.store(state, action, reward, next_state, terminated)
            ## Flip states
            state = next_state
            ## Termination check
            if terminated:
                agent.alighn_target_model()
                break
            ## refer to experience replay for retraining
            if len(agent.experience_replay) > batch_size:
                agent.retrain(batch_size)
            ## update the bar
            if timestep%10 == 0:
                bar.update(timestep/10 + 1)

    def play_multiple_episodes(self, env, agent, num_episodes, bar, timesteps_per_episode, batch_size):
        for e in range(0, num_episodes):
            ## reset the environment
            state = env.reset()
            state = np.reshape(state, [1, 1])
            ## initialize the variables
            reward = 0
            terminated = False
            self.play_one_episode(env, state, bar, agent, timesteps_per_episode, batch_size)
            bar.finish()
            if (e + 1) % 10 == 0:
                print("**********************************")
            print("Episode: {}".format(e + 1))
            env.render()
            print("**********************************")


if __name__ == '__main__':
    ## Creating the Taxi v2 environment
    env = gym.make("Taxi-v2").env
    ## Defining the adam optimizer with the learning rate
    optimizer = Adam(learning_rate=0.01)
    ## Creating a learning agent using the environment and the optimizer
    agent_7bb_booty = LearningAgent(env, optimizer)
    ## Define some basic parameters
    batch_size = 32
    num_episodes = 100
    timesteps_per_episode = 1000
    ## Give a summary of the DQN
    agent_7bb_booty.q_network.summary()
    ## defining the progress bar
    bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    ## Let the agent play! 7bb play bibibibi play!
    agent_7bb_booty.play_multiple_episodes(env, agent_7bb_booty, num_episodes, bar, timesteps_per_episode, batch_size)
    # print(f'Number of states: {env.observation_space.n}')
    # print(f'Number of actions: {env.action_space.n}')


