from __future__ import print_function, division
from builtins import range

import numpy as np
import sys 
import os 
import gym
import matplotlib.pyplot as plt 
from datetime import datetime 
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler 
from sklearn.kernel_approximation import RBFSampler 
from self_q_learning_bins import plot_running_avg


class SGDRegressor:
    def __init__(self,D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 0.1

    def partial_fit(self, X, Y):
        return X.dot(self.w)

    def predict(self, X):
        return X.dot(self.w)

class FeatureTransformer:
    def __init__(self,env):
        ## note that the state samples are poor, b/c you get 
        ## velocities that are close to infinity
        observation_examples = np.random.random((2000,4))*2 - 1
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        ## convert a state into a feature representation
        ## RBF kernels with different variances are used to cover the different
        ## parts of the space

        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=1000)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=1000))
            ])

        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))
        
        self.dimensions = feature_examples.shape[1]
        self.scaler = scaler 
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)

# Holds one SGDRegressor for each action
class Model: 
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(feature_transformer.dimensions)
            self.models.append(model)

    def predict(self,s):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        result = np.stack([m.predict(X) for m in self.models]).T 
        return result 

    def update(self,s,a,G):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s ,eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(env, model, eps, gamma, render):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 2000:
        if render == True:
            env.render()
        # if we reach 2000, just quit, don't want this going forever
        # the 200 limit seems a bit early
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        if done:
            reward = -200

        ## update the model
        next = model.predict(observation)
        assert(next.shape == (1, env.action_space.n))
        G = reward + gamma*np.max(next)
        model.update(prev_observation, action, G)

        if reward == 1:
            totalreward += reward
        iters += 1
    if render == True:
        env.close()

    return totalreward


def main():
    gym.envs.register(
        id='Cart-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=10000
        #reward_threshold=-110.0,
    )
    env = gym.make('Cart-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.99 

    # if 'monitor' in sys.argv:
    #     filename = os.path.basename(__file__).split('.')[0]
    #     monitor_dir = './' + filename + '_' + str(datetime.now())
    #     env = wrappers.Monitor(env, monitor_dir)
        
    N = 500
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N): 
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(env, model, eps, gamma, False)
        totalrewards[n] = totalreward

        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())
    print(f"Average reward for last 100 episodes: {totalrewards[-100:].mean()}")
    print("Total Steps: ", totalrewards.sum())


    play_one(env, model, eps, gamma, True)

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()
    plot_running_avg(totalrewards)

if __name__ == '__main__':
    main()
