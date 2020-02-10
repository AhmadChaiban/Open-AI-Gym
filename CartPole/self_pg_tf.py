import gym
import numpy as np
import tensorflow as tf
from self_q_learning import plot_running_avg

## Model object
class Model:
    def __init__(self, input_dim, dense1, dense2, lr, from_logits):
        ## Defining the model
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(dense1, input_dim = input_dim, activation = 'relu'))
        self.model.add(tf.keras.layers.Dense(dense2, activation = 'softmax'))
        self.model.build()
        ## Defining the optimizer and loss function
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        self.compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= from_logits)
        ## Variable to hold gradients
        self.gradBuffer = self.model.trainable_variables
        for index,grad in enumerate(self.gradBuffer):
            self.gradBuffer[index] = grad*0

    def discount_rewards(self, rewards, gamma=0.8):
        # gamma: discount rate
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def get_action(self, logits):
        a_dist = logits.numpy()
        # Choose random action with p = action dist
        action = np.random.choice(a_dist[0],p=a_dist[0])
        action = np.argmax(a_dist == action)
        return action

    def apply_gradient(self, ep_memory):
        for grads, r in ep_memory:
            for ix,grad in enumerate(grads):
                self.gradBuffer[ix] += grad * r

    def play_one_episode(self, env, ep_score, ep_memory, update_every, render):
            ## reset the environment
            observation = env.reset()
            done = False
            while not done:
                if render == True:
                    env.render()
                observation = observation.reshape([1,len(observation)])
                with tf.GradientTape() as tape:
                    #forward pass
                    logits = self.model(observation)
                    action = self.get_action(logits)
                    loss = self.compute_loss([action], logits)
                # make the choosen action
                observation, reward, done, _ = env.step(action)
                ep_score += reward
                if done:
                    reward = -200 # small trick to make training faster
                grads = tape.gradient(loss, self.model.trainable_variables)
                ep_memory.append([grads,reward])
            #Discound the rewards
            if render == True:
                env.close()
            ep_memory = np.array(ep_memory)
            ep_memory[:,1] = self.discount_rewards(ep_memory[:,1])
            self.apply_gradient(ep_memory)
            if e % update_every == 0:
                self.optimizer.apply_gradients(zip(self.gradBuffer, self.model.trainable_variables))
                for ix,grad in enumerate(self.gradBuffer):
                    self.gradBuffer[ix] = grad * 0

            return ep_memory, ep_score

if __name__ == '__main__':
    gym.envs.register(
        id='Cart-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=10000
        #reward_threshold=-110.0,
    )
    env = gym.make('Cart-v0')

    model = Model(4, 32, 2, 0.01, True)

    N = 1000
    scores = []
    update_every = 5
    scores = []
    for e in range(N):
        ep_memory = []
        ep_score = 0
        ep_memory, ep_score  = model.play_one_episode(env, ep_score, ep_memory, update_every, False)
        scores.append(ep_score)
        if e % 100 == 0:
            print(f"Episode  {e+1}  Score  {np.mean(scores[-100:])}")

    ep_memory = []
    ep_score = 0
    plot_running_avg(np.array(scores))
    N_render = 10
    for _ in range(N_render):
        model.play_one_episode(env, ep_score, ep_memory, update_every, True)
    print(f"Episode {N+1} Score {scores[-1]}")