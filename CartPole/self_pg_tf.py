import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from self_q_learning_bins import plot_running_avg

## Creating an object to test different architectures
class HiddenLayer:
    def __init__(self, M1, M2, f = tf.nn.tanh, use_bias = True):
        self.W = tf.Variable(tf.random_normal(shape = M1, M2))
