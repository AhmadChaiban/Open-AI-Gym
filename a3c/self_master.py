import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import optimizers
from self_networks import ActorCriticModel
from self_random_agent import RandomAgent
import multiprocessing
import matplotlib.pyplot as plt
from self_worker import Worker
import os

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

    def worker_assignment(self, res_queue):
        workers = [Worker(self.state_size,
                                 self.action_size,
                                 self.global_model,
                                 self.optimizer, res_queue,
                                 i, game_name=self.game_name) for i in range(multiprocessing.cpu_count())]
        return workers

    def collect_moving_average_rewards(self, res_queue):
        # record episode reward to plot
        moving_average_rewards = []
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        return moving_average_rewards

    def plot_moving_average_rewards(self, moving_average_rewards):
        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.show()

    def train(self, maxEpisodes):
        random_agent = RandomAgent(self.game_name, maxEpisodes)
        random_agent.run()

        res_queue = Queue()

        workers = self.worker_assignment(res_queue)

        for i, worker in enumerate(workers):
            print(f"Starting worker {i}")
            worker.start()

        moving_average_rewards = self.collect_moving_average_rewards(res_queue)

        [w.join() for w in workers]

        self.plot_moving_average_rewards(moving_average_rewards)

    def play(self):
        env = gym.make(self.game_name).unwrapped
        state = env.reset()
        model = self.global_model
        model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name))
        print('Loading model from: {}'.format(model_path))
        model.load_weights(model_path)
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                env.render(mode='rgb_array')
                policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                policy = tf.nn.softmax(policy)
                action = np.argmax(policy)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()







