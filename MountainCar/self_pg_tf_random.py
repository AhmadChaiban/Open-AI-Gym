import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

## Need a gaussian activation function 
class Model:
    def __init__ (self, input, dense1, dense2):
        self.model = Sequential([
            Dense(dense1, input_dim = input, )
        ])



if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')

