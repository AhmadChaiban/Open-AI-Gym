## The Open AI gym
import gym
## Wrappers save videos
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

## Decides whether to do the action or not 
def get_action(observation,weights):
    if observation.dot(weights) > 0:
        return 1
    else:
        return 0
## Plays a single episode  
def play_episode(env,observation,weights):
    env.reset()
    done = False
    frames = []
    while not done:
        env.render()
        frames.append(env.render(mode = 'rgb_array'))
        action = get_action(observation,weights)
        observation,reward,done,info = env.step(action)
    env.close()
    return observation,reward,done,info,frames


## Random search algorithm begins here
## This is my own personal attempt at the random search algorithm
env = gym.make('CartPole-v0')
## resetting the environment 
env.reset()
## getting a random obersvation from a step to initialize the variable
observation,reward,done,info = env.step(env.action_space.sample())
## Setting some random weights
weights = [1,1,1,1]
## Setting the number of times to run random search
epochs = 10
## Number of episodes per random search
episodes = 20

## This function applies the random search, it takes the 
## gym enivronment and applies a roandom search to it 
## to try to randomly find the best play. 
def random_search(env,observation,weights,epochs,episodes):
    ## this will hold the number of actions per episode
    forMax_epoch_list = []
    ## this value will eventually hold the maximum number 
    ## of actions per episode, it is first set to 0
    prev_epoch_list_max = 0
    ## Looping the algorithm
    for epoch in range(epochs):
        ## Setting random new weights to be tested on
        new_weights = np.random.random(4)*2 -1
        ## looping per episode 
        for episode in range(episodes):
            env.reset()
            done = False
            ## Keeps track of how much an episode was played
            epoch_counter = 0
            ## Playing the episode
            while not done:
                action = get_action(observation, weights)
                observation,reward,done,info = env.step(action)
                epoch_counter +=1
            ## appending each counter per episode
            forMax_epoch_list.append(epoch_counter)
            ## Providing feeback on cmd
            print(f"Epoch: {epoch} | Episode: {episode} | Maximum: {prev_epoch_list_max}")
        ## comparing to see if it did better this time
        if np.mean(forMax_epoch_list) > prev_epoch_list_max:
            ## setting the new weights if this run of the algorithm was better
            weights = new_weights
            ## setting a new high score 
            prev_epoch_list_max = max(forMax_epoch_list)
        print(f"final Maximum: {prev_epoch_list_max}")
    return observation, weights

## Random search!
observation,weights = random_search(env,observation,weights,epochs,episodes)
## Play the final episode here
observation,reward,done,info,frames = play_episode(env,observation,weights)