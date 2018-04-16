import numpy as np

class AbstractAgent(object):
    def __init__(self, config, n_action):
        self.n_action = n_action

    def forward(self, n_action):
        return np.random.randint(0,self.n_action)

    def optimize(self):
        pass

    def apply_config(config):
        # This could be useful to allow agent-specifig config steps 
        pass

    def callback(epoch):
        # Allows for unified training API : all specific action in-between
        # epochs are done here
        pass
