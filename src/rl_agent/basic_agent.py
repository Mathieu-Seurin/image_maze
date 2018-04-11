import numpy as np

class AbstractAgent(object):
    def __init__(self, config, n_action):
        self.n_action = n_action
    def forward(self, n_action):
        return np.random.randint(0,self.n_action)
    def optimize(self):
        pass