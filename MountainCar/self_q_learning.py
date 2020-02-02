from __future__ import print_function,division
from builtins import range

import gym
import os 
import sys
import numpy as np 
import matplotlib.pyplot as plt 
from gym import wrappers
from datetime import datetime 
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

## SGD Regressor defaults
## loss = 'squared_loss', penalty = 'l2', alpha = 0.0001,
## l1_ratio = 0.15, fit_intercept = True, n_iter = 5, shuffle = True,
## verbose = 0, epsilon = 0.1, random_state = None, learning_rate = 'inv_scaling',
## eta0 = 0.01, power_t = 0.25, warm_start = False, average = False

class FeatureTransformer:
    def __init__ (self,env):
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        ## Standardize the observations so we have mean 0 and variance 1
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        ## Converts the state into a feature representations
        ## We use the rbf kernel with different variances to cover different parts of the space
        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma = 5.0, n_components=500)),
            ('rbf2', RBFSampler(gamma = 2.0, n_components=500)),
            ('rbf3', RBFSampler(gamma = 1.0, n_components=500)),
            ('rbf4', RBFSampler(gamma = 0.5, n_components=500))
        ])
        featurizer.fit(scaler.transform(observation_examples))

        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self,observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)

## Holds one SGDRegressor for each action
class Model:
    def __init__(self, env, feature_transformer,learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate = learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]),[0])
            self.models.append(model)
    
    def predict(self,s):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape)==2)
        return np.array([m.predict(X)[0] for m in self.models])

    def update(self,s,a,G):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape)==2)
        self.models[a].partial_fit(X,[G])

    def sample_action(self,s,eps):
        ## Sampling actions with epsilon greedy
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

## Returns a list of states and rewards, and returns the total reward
def play_one(model,env,eps,gamma,render):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0 
    while not done and iters < 10000:
        if render == True:
            env.render()
        action = model.sample_action(observation,eps)
        prev_observation = observation
        observation,reward,done,info = env.step(action)

        ## updating the model 
        G = reward + gamma*np.max(model.predict(observation)[0])
        model.update(prev_observation,action,G)
        
        totalreward += reward
        iters += 1 
    if render == True:
        env.close()
    return totalreward

def plot_cost_to_go(env, estimator, num_tiles=20):
  x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
  y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
  X, Y = np.meshgrid(x, y)
  # both X and Y will be of shape (num_tiles, num_tiles)
  Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
  # Z will also be of shape (num_tiles, num_tiles)

  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, Y, Z,
    rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
  ax.set_xlabel('Position')
  ax.set_ylabel('Velocity')
  ax.set_zlabel('Cost-To-Go == -V(s)')
  ax.set_title("Cost-To-Go Function")
  fig.colorbar(surf)
  plt.show()


def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()


def main(show_plots=True):
    gym.envs.register(
        id='Mountain-v0',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=10000
        #reward_threshold=-110.0,
    )
    env = gym.make('Mountain-v0')
    ft = FeatureTransformer(env)
    model = Model(env,ft,'constant')
    gamma = 0.99

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 300
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        if n == 199:
            print(f"eps: {eps}")
        totalreward = play_one(model,env,eps,gamma,False)
        totalrewards[n] = totalreward

        if (n + 1)%100 == 0:
            print(f"Episode: {n} Total reward: {totalreward}")
        print(f"Average reward for last 100 episodes: {totalrewards[-100:].mean()}")
        print(f"Total Steps: {-totalrewards.sum()}")

        play_one(model,env,eps,gamma,True)

        if show_plots:
            plt.plot(totalrewards)
            plt.title("Rewards")
            plt.show()

            plot_running_avg(totalrewards)

            plot_cost_to_go(env,model)

if __name__ == '__main__':
    # for i in range(10):
    #   main(show_plots=False)
    main()
