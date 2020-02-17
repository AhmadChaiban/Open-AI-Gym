import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from queue import Queue


## This is the random agent that will play the specified game
class RandomAgent:
    def __init__(self, envName, maxEps):
        self.env = gym.make(envName)
        self.maxEpisodes = maxEps
        self.global_moving_average_reward = 0
        self.res_queue = Queue()

    def run(self):
        reward_avg = 0
        for episode in range(self.maxEpisodes):
            reward_avg += self.play_one_episode(episode)
        final_avg = reward_avg / float(self.maxEpisodes)
        print(f"Average score across {self.maxEpisodes} episodes: {final_avg}")
        return final_avg

    def play_one_episode(self, episode):
        done = False
        self.env.reset()
        reward_sum = 0.0
        steps = 0
        while not done:
            # sample randomly from the action space and step
            _, reward, done, _ = self.env.step(self.env.action_space.sample())
            steps += 1
            reward_sum += reward
        ## Record some Statistics
        self.global_moving_average_reward = record(episode,
                                                   reward_sum,
                                                   0,
                                                   self.global_moving_average_reward,
                                                   self.res_queue, 0, steps)

        return reward_sum
