import torch
class Config:
    def __init__(self):
        self.buffer_size = 50000        # replay buffer size
        self.batch_size = 256           # minibatch size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99               # discount factor
        self.tau = .001                  # for soft update of target parameters
        self.LR_actor = 1e-4            # learning rate of the actor
        self.LR_critic = 3e-4           # learning rate of the critic
        self.weight_decay = 0.0         # L2 weight decay
        self.max_episodes = 2500
        self.epsilon_decay = .1**(1/(2500 + 300))    # the 300 more or less adjusts for the delay in start of training
        print('ep decay:', self.epsilon_decay)
        self.fc1_units = 256
        self.fc2_units = 128
        self.sigma = 0.2                # noise variance
        self.num_repeats = 3            # repeat learning step x times
        self.seed = None
        self.trial = 0

    def __setattr__(self, key, value):
        self.__dict__[key] = value
