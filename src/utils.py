import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import random
from gym.envs.classic_control import rendering
import logging

# GPU compatibility setup
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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


def write_json_config_file(filename='test', conv_shapes=[5,], dense_shapes=[],
                            use_batch_norm=False, n_out=2,
                            input_resolution=(64, 64), n_channels=3, batch_size=32):
    params = {
        'conv_shapes':conv_shapes,
        'dense_shapes':dense_shapes, #Not implemented yet
        'use_batch_norm':use_batch_norm,
        'n_out':n_out,
        'input_resolution':input_resolution,
        'n_channels':n_channels,
        'batch_size':batch_size,
    }

    with open('./trained_agents/'+filename+'.cfg', 'w') as outfile:
        json.dump(params, outfile)

resize = T.Compose([T.ToPILImage(),T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])

viewer = rendering.SimpleImageViewer()
def get_screen(env, display=False):
    screen = env.render(mode='rgb_array')  # transpose into torch order (CHW)
    if display:
        viewer.imshow(screen)
    screen = screen.transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    # print(screen[:5, :5])
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).type(Tensor)
