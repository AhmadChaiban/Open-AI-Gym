import numpy as np
import pandas as pd
import gym 
from gym import wrappers
import matplotlib.pyplot as plt 
import os 
import sys
from datetime import datetime

## Build state function
## rewrite the lambda notation at some point 
def build_state(features):
    stateBuild = ''
    for i in range(len(features)):
        stateBuild += str(int(features[i]))
    return int(stateBuild)
## Need to put the right comment here. Understand this before proceeding
## Check out the documentation of np.digitize
def to_bin(value,bins):
    return np.digitize(x=[value],bins= bins)[0]

class FeatureTransformer:
    def __init__(self):
        ## In order to improve this, histograms should be plotted to check the 
        ## frequency of each bin.
        ## We're just trying random values for now
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)

    def transform(self,observation):
        cart_position,cart_velocity,pole_angle,pole_velocity = observation
        stateBuild = build_state([
            to_bin(cart_position,self.cart_position_bins),
            to_bin(cart_velocity,self.cart_velocity_bins),
            to_bin(pole_angle,self.pole_angle_bins),
            to_bin(pole_velocity,self.pole_velocity_bins)
        ])
        return stateBuild
    
class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer

        num_states = 10**env.observation_space.shape[0]
        num_actions = env.action_space.n 
        self.Q = np.random.uniform(low=-1,high=1,size=(num_states,num_actions))

    def predict(self,s):
        x = self.feature_transformer.transform(s)
        return self.Q[x]

    def update(self,s,a,G):
        x = self.feature_transformer.transform(s)
        self.Q[x,a] += 1e-2*(G - self.Q[x,a])
    
    def sample_action(self,s,eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            p = self.predict(s)
            return np.argmax(p)

def play_one(model,eps,gamma,render):
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

        totalreward += reward

        if done and iters < 9999:
            reward = -300
        G = reward + gamma*np.max(model.predict(observation))
        model.update(prev_observation,action,G)

        iters += 1
    if render == True:
        env.close()
    return totalreward

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0,t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

if __name__ == '__main__':
    gym.envs.register(
        id='Cart-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=10000
        #reward_threshold=-110.0,
    )
    env = gym.make('Cart-v0')
    ft = FeatureTransformer()
    model = Model(env,ft)
    gamma = 0.9

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)
    N = 10000
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        if n == 9999:
            totalreward = play_one(model, eps, gamma,True)
        else:
            totalreward = play_one(model,eps,gamma,False)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print(f"The final reward is: {totalrewards[-1]}")
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)
        



