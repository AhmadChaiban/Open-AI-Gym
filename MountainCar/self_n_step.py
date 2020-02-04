from __future__ import print_function, division 
from builtins import range 

## Let's adapt the q learning script to use the N-step method instead

import gym 
import os 
import sys
import numpy as np 
import matplotlib.pyplot as plt 
from gym import wrappers
from datetime import datetime

import self_q_learning 
from self_q_learning import plot_cost_to_go, FeatureTransformer, Model, plot_running_avg
from self_q_learning import play_one as play_one_v2

class SGDRegressor: 
    def __init__ (self, **kwargs):
        self.w = None 
        self.lr = 1e-2

    def partial_fit(self, X, Y):
        if self.w is None: 
            D = X.shape[1]
            self.w = np.random.randn(D)/np.sqrt(D)
        self.w += self.lr*(Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)

self_q_learning.SGDRegressor = SGDRegressor

# calculate everything up to max[Q(s,a)]
# Ex.
# R(t) + gamma*R(t+1) + ... + (gamma^(n-1))*R(t+n-1) + (gamma^n)*max[Q(s(t+n), a(t+n))]
# def calculate_return_before_prediction(rewards, gamma):
#   ret = 0
#   for r in reversed(rewards[1:]):
#     ret += r + gamma*ret
#   ret += rewards[0]
#   return ret

## Returns a list of states and rewards, and the total reward
def play_one(model, eps, gamma, n=5):
    observation = env.reset()
    done = False
    totalreward = 0
    rewards = []
    states = []
    actions = []
    iters = 0
    ## array of [gamma^0, gamma^1, ....., gamma^(n-1)]
    multiplier = np.array([gamma]*n)**np.arange(n)
    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        states.append(observation)
        actions.append(action)

        prev_observation = observation 
        observation, reward, done, info = env.step(action)

        rewards.append(reward)

        ## update the model 
        if len(rewards) >= n:
            # return_up_to_prediction = calculate_return_before_prediction(rewards, gamma)
            return_up_to_prediction = multiplier.dot(rewards[-n:])
            G = return_up_to_prediction + (gamma**n)*np.max(model.predict(observation)[0])
            model.update(states[-n], actions[-n], G)

        # if len(rewards) > n:
        #   rewards.pop(0)
        #   states.pop(0)
        #   actions.pop(0)
        # assert(len(rewards) <= n)

        totalreward += reward
        iters+=1 

    #empty the cache
    if n==1: 
        rewards = []
        states = []
        actions = []
    else:
        rewards = rewards[-n+1:]
        states = states[-n+1:]
        actions = actions[-n+1:]
    ## The goal position of mountain car is 0.5
    ## This is if the Agent makes it to the goal
    if observation[0] >= 0.5:
        while len(rewards) > 0:
            G = multiplier[:len(rewards)].dot(rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    else: 
        ## if the Agent didn't make it to the goal
        while len(rewards) > 0:
            guess_rewards = rewards + [-1]*(n - len(rewards))
            G = multiplier.dot(guess_rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    return totalreward

if __name__ == '__main__':
    gym.envs.register(
        id='Mountain-v0',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=10000
        #reward_threshold=-110.0,
    )
    env = gym.make('Mountain-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, 'constant')
    gamma = 0.99

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 300
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        print("episode:", n, "total reward:", totalreward)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()
    
    plot_running_avg(totalrewards)
    play_one_v2(model, env, eps, gamma, True)