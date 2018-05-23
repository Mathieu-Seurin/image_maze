import numpy as np

class AbstractAgent(object):
    # Template of all methods used in the rest of the algorithms
    def __init__(self, config, n_action):
        self.n_action = n_action

    def forward(self, n_action, *args, **kwargs):
        return np.random.randint(0, self.n_action)

    def optimize(self, *args, **kwargs):
        pass

    def apply_config(self, config):
        # This could be useful to allow agent-specifig config steps
        pass

    def callback(self, epoch):
        # Allows for unified training API : all specific action in-between
        # epochs are done here
        pass

    def save_state(self, folder):
        # Store the whole agent state somewhere
        pass

    def load_state(self, folder):
        # Retrieve the whole agent state somewhere
        pass

    def eval(self):
        pass

    def train(self):
        pass
