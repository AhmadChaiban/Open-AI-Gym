## The Open AI gym`
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
def play_episode(weights):
    observation = env.reset()
    done = False
    frames = []
    while not done:
        env.render()
        action = get_action(observation,weights)
        observation,reward,done,info = env.step(action)
    env.close()
    return observation,reward,done,info,frames


## Random search algorithm begins here
## This is my own personal attempt at the random search algorithm
env = gym.make('CartPole-v0')
## Creating a new cartpole environment with a different limit 
gym.envs.register(
    id='Cart-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=10000
    #reward_threshold=-110.0,
)
env = gym.make('Cart-v0')
## resetting the environment 
observation = env.reset()
## Setting some random weights
weights = [0.1,0.1,0.1,0.1]
## Setting the number of times to run random search
epochs = 10
## Number of episodes per random search
episodes = 10

## This function applies the random search, it takes the 
## gym enivronment and applies a roandom search to it 
## to try to randomly find the best play. 
def random_search(env,observation,weights,epochs,episodes):
    ## Overarching average length per random search iteration
    avg_per_iteration = []
    ## this value will eventually hold the maximum number 
    ## of actions per episode, it is first set to 0
    prev_epoch_list_max = 0
    ## Looping the algorithm
    for epoch in range(epochs):
        ## Setting random new weights to be tested on
        new_weights = np.random.random(4)*2 -1
        ## this will hold the number of actions per episode
        forMax_epoch_list = []
        ## looping per episode 
        for episode in range(episodes):
            observation = env.reset()
            done = False
            ## Keeps track of how much an episode was played
            epoch_counter = 0
            ## Playing the episode
            while not done:
               # env.render()
                action = get_action(observation, new_weights)
                observation,reward,done,info = env.step(action)
                epoch_counter +=1
            #env.close()
            ## appending each counter per episode
            forMax_epoch_list.append(epoch_counter)
            ## Providing feeback on cmd
            print(f"Epoch: {epoch+1} | Episode: {episode+1} | Maximum: {prev_epoch_list_max} | Score: {epoch_counter}")
        ## comparing to see if it did better this time
        avg_per_iteration.append(np.mean(forMax_epoch_list))
        if np.mean(forMax_epoch_list) > prev_epoch_list_max:
            ## setting the new weights if this run of the algorithm was better
            weights = new_weights
            ## setting a new high score 
            prev_epoch_list_max = max(forMax_epoch_list)
        print(f"final Maximum: {prev_epoch_list_max}")
    return weights, avg_per_iteration

## Random search!
weights,avg_per_iteration = random_search(env,observation,weights,epochs,episodes)
## Play the final episode here
observation,reward,done,info,frames = play_episode(weights)
## Plotting the average score per random search iteration
plt.plot(avg_per_iteration)
plt.show()
## Find a way to pick out the best observation