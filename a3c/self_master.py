import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import optimizers
from self_networks import ActorCriticModel
# from self_worker import Worker

class MasterAgent:
    def __init__(self):
        self.game_name = 'CartPole-v0'

        env = gym.make(self.game_name)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.optimizer = optimizers.Adam(learning_rate = 0.01)
        print(f"State Size {self.state_size} Action Size {self.action_size}")

        self.global_model = ActorCriticModel(self.state_size, self.action_size)
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

    def worker_assignment(self):
        return workers = [Worker(self.state_size,
                                 self.action_size,
                                 self.global_model,
                                 self.opt, res_queue,
                                 i, game_name=self.game_name,
                                 save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]


    # def train(self):



