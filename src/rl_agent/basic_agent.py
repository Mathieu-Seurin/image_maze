import numpy as np

class AbstractAgent(object):
    # Template of all methods used in the rest of the algorithms
    def __init__(self, config, n_action):
        self.n_action = n_action

    def forward(self, *args, **kwargs):
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

    def save_state(self):
        # Store the whole agent state somewhere
        return None, None

    def load_state(self, dict_state, memory=None):
        # Retrieve the whole agent state somewhere
        pass

    def eval(self):
        pass

    def train(self):
        pass


class PerfectAgent(AbstractAgent):
    def __init__(self, config, n_action):
        super(PerfectAgent, self).__init__(config, n_action)

    def forward(self, state, *args, **kwargs):

        reward_position = state['info']['reward_position']
        state_position = state['info']['agent_position']

        diff_x = reward_position[0] - state_position[0]
        diff_y = reward_position[1] - state_position[1]

        action = None

        if diff_x > 0:
            action = 0 # North
        elif diff_x < 0:
            action = 1 # South

        if diff_y > 0:
            action = 3 # West
        elif diff_y < 0:
            action = 2 # East

        assert action is not None, "Perfect agent is not so perfect."

        return action

