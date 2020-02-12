import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

## Need a gaussian activation function
class PolicyModel:
    def __init__ (self, input, dense1, dense2):
        self.model = Sequential([
            Dense(dense1, input_dim = input, activation = 'relu'),
            Dense(dense2, activation = tf.math.softplus)
        ])

        ## To model the mean and the variance
        self.mean_layers = []
        self.var_layers = []
        ## Gather the parameters
        self.params = []

        ## Does a gaussian activation with the random search
        ## optimization do the trick? Check!
    #
    # def gaussian_activation(self, x):
    #     sq = tf.square(x)
    #     neg = tf.negative(sq)
    #     return tf.exp(neg)


def random_search(env, model, gamma):
    totalrewards = []
    best_avg_totalreward = float('-inf')
    best_model = model
    num_episodes_per_param_test = 3
    for t in range(100):
        tmp_model = best_model.copy()
        tmp_model.perturb_params()

        avg_totalrewards = play_multiple_episodes(
            env,
            num_episodes_per_param_test,
            tmp_model,
            gamma
        )
        totalrewards.append(avg_totalrewards)

        if avg_totalrewards > best_avg_totalreward:
            best_model = tmp_model
            best_avg_totalreward = avg_totalrewards
    return totalrewards, best_model


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    ## Build the model here
    gamma = 0.99

    totalrewards, pmodel = random_search(env, model, gamma)
    print(f'max reward: {np.max(totalrewards)}')

    avg_totalrewards = play_multiple_episodes(env, 100, pmodel, gamma, print_iters=True)
    print("avg reward over 100 episodes with best models:", avg_totalrewards)

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()


