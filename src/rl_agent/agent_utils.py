import numpy as np
from collections import namedtuple

import torch.nn as nn
import random
import logging


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Flatten(nn.Module):
  def forward(self, x):
    N, C, H, W = x.size() # read in N, C, H, W
    return x.view(N, -1)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = np.min([batch_size, len(self.memory)])
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def freeze_as_np_dict(tensor_dict):
    out = {}
    for key in tensor_dict.keys():
        out[key] = tensor_dict[key].cpu().clone().numpy()
    return out

def check_params_changed(dict1, dict2):
    "Takes two parameters dict (key:torchtensor) and prints a warning if identical"
    for key in dict1.keys():
        tmp1 = dict1[key]
        tmp2 = dict2[key]
        if np.max(np.abs(tmp1 - tmp2))==0:
            logging.warning('No change in params {}'.format(key))
