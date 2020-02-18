import threading
from self_networks import ActorCriticModel
import gym

class Worker(threading.Thread):
    # Setting up global variables across different threads
    global_episode = 0
    best_score = 0
    save_lock = threading.lock()

    def __init__(self,
                 state_size,
                 action_size,
                 global_model,
                 optimizer,
                 result_queue,
                 idx,
                 game_name = 'CartPole-v0'):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.global_model = global_model
        self.result_queue = result_queue
        self.optimizer = optimizer
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.worker_idx = idx
        self.game_name = game_name
        self.env = gym.make(self.game_name)
        self.eps_loss = 0.0