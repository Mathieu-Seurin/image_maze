import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from rl_agent.film_utils import init_modules

from .gpu_utils import FloatTensor


class FiLM(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def forward(self, x, gammas, betas):
    gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
    betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
    return (gammas * x) + betas


class FiLMedResBlock(nn.Module):
  def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True,
               with_cond=[False], dropout=0, num_extra_channels=0, extra_channel_freq=1,
               with_input_proj=0, num_cond_maps=0, kernel_size=3, batchnorm_affine=False,
               num_layers=1, condition_method='bn-film', debug_every=float('inf')):


    if out_dim is None:
      out_dim = in_dim
    super(FiLMedResBlock, self).__init__()
    self.with_residual = with_residual
    self.with_batchnorm = with_batchnorm
    self.with_cond = with_cond
    self.dropout = dropout
    self.extra_channel_freq = 0 if num_extra_channels == 0 else extra_channel_freq
    self.with_input_proj = with_input_proj  # Kernel size of input projection
    self.num_cond_maps = num_cond_maps
    self.kernel_size = kernel_size
    self.batchnorm_affine = batchnorm_affine
    self.num_layers = num_layers
    self.condition_method = condition_method
    self.debug_every = debug_every

    # if self.with_input_proj % 2 == 0:
    #   raise(NotImplementedError)
    if self.kernel_size % 2 == 0:
      raise(NotImplementedError)
    if self.num_layers >= 2:
      raise(NotImplementedError)

    if self.condition_method == 'block-input-film' and self.with_cond[0]:
      self.film = FiLM()
    if self.with_input_proj:
      self.input_proj = nn.Conv2d(in_dim + (num_extra_channels if self.extra_channel_freq >= 1 else 0),
                                  in_dim, kernel_size=self.with_input_proj, padding=self.with_input_proj // 2)

    num_extra_channels = num_extra_channels if self.extra_channel_freq >= 2 else 0
    self.conv1 = nn.Conv2d(in_dim + self.num_cond_maps + num_extra_channels,
                            out_dim, kernel_size=self.kernel_size,
                            padding=self.kernel_size // 2)

    if self.condition_method == 'conv-film' and self.with_cond[0]:
      self.film = FiLM()
    if self.with_batchnorm:
      self.bn1 = nn.BatchNorm2d(out_dim, affine=((not self.with_cond[0]) or self.batchnorm_affine))
    if self.condition_method == 'bn-film' and self.with_cond[0]:
      self.film = FiLM()
    if dropout > 0:
      self.drop = nn.Dropout2d(p=self.dropout)
    if ((self.condition_method == 'relu-film' or self.condition_method == 'block-output-film')
         and self.with_cond[0]):
      self.film = FiLM()

    init_modules(self.modules())

  def forward(self, x, gammas=None, betas=None, extra_channels=None, cond_maps=None):

    if self.condition_method == 'block-input-film' and self.with_cond[0]:
      x = self.film(x, gammas, betas)

    # ResBlock input projection
    if self.with_input_proj:
      if extra_channels is not None and self.extra_channel_freq >= 1:
        x = torch.cat([x, extra_channels], 1)
      x = F.relu(self.input_proj(x))
    out = x

    # ResBlock body
    if cond_maps is not None:
      out = torch.cat([out, cond_maps], 1)
    if extra_channels is not None and self.extra_channel_freq >= 2:
      out = torch.cat([out, extra_channels], 1)
    out = self.conv1(out)
    if self.condition_method == 'conv-film' and self.with_cond[0]:
      out = self.film(out, gammas, betas)
    if self.with_batchnorm:
      out = self.bn1(out)
    if self.condition_method == 'bn-film' and self.with_cond[0]:
      out = self.film(out, gammas, betas)
    if self.dropout > 0:
      out = self.drop(out)
    out = F.relu(out)
    if self.condition_method == 'relu-film' and self.with_cond[0]:
      out = self.film(out, gammas, betas)

    # ResBlock remainder
    if self.with_residual:
      out = x + out
    if self.condition_method == 'block-output-film' and self.with_cond[0]:
      out = self.film(out, gammas, betas)

    return out

class VisionFilmGen(nn.Module):

    def __init__(self, config):
        super(VisionFilmGen, self).__init__()

        self.num_features_per_block = 15
        self.num_block_to_modulate = 2
        self.num_features_to_modulate = self.num_block_to_modulate*self.num_features_per_block

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc_gammas = nn.Linear(7 * 7 * 32, self.num_features_to_modulate)
        self.fc_betas = nn.Linear(7 * 7 * 32, self.num_features_to_modulate)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)

        gammas = self.fc_gammas(out)
        betas = self.fc_betas(out)
        return gammas, betas


class FilmedNet(nn.Module):
    def __init__(self, config, n_actions, state_dim, is_multi_objective):
        super(FilmedNet, self).__init__()

        self.lr = config["learning_rate"]
        self.gamma = config["gamma"]


        self.n_resblocks = config["n_resblock"]
        self.n_hidden = config["n_hidden"]

        self.use_film = config["use_film"]

        self.in_dim = state_dim[0] # Number of channels
        self.is_multi_objective = is_multi_objective
        if is_multi_objective and self.use_film:
            # Todo : Check that image is always 3 channel
            self.in_dim += 3
            self.film_gen = VisionFilmGen(config=config)

        self.size_img = state_dim[1]*state_dim[2]
        self.out_channel = 1
        self.n_actions = n_actions

        self.resblocks = nn.ModuleList()
        for num_resblock in range(self.n_resblocks):
            current_resblock = FiLMedResBlock(in_dim=self.in_dim,
                                              with_residual=True,
                                              with_batchnorm=False,
                                              with_cond=[True])

            self.resblocks.append(current_resblock)


        # head
        self.head_conv = nn.Conv2d(in_channels=self.in_dim,
                                   out_channels=1,
                                   kernel_size=1)

        # Todo : attention head ?

        self.fc1 = nn.Linear(in_features=self.size_img * self.out_channel, out_features=self.n_hidden)
        self.fc2 = nn.Linear(in_features=self.n_hidden, out_features=self.n_actions)


        if config['optimizer'] == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        elif config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            assert False, 'Optimizer not recognized'

    def forward(self, x):

        batch_size = x['env_state'].size(0)

        if self.use_film:
            gammas, betas = self.film_gen.forward(x['objective'])
            x = x['env_state']
        else:

            if self.is_multi_objective :
                x = torch.cat((x['env_state'], x['objective']), dim=1)
            else:
                x = x['env_state']

            # Gammas = all ones
            # Betas = all zeros
            gammas = Variable(torch.ones(batch_size, self.in_dim*self.n_resblocks).type(FloatTensor))
            betas = Variable(torch.zeros_like(gammas.data).type(FloatTensor))

        for i,resblock in enumerate(self.resblocks):
            gamma_beta_id = slice(self.in_dim*i,self.in_dim*(i+1))
            x = resblock.forward(x, gammas=gammas[:, gamma_beta_id], betas=betas[:, gamma_beta_id])

        x = self.head_conv(x)
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    import numpy as np
    from torch.autograd import Variable
    config = {
        "num_layer" : 2,
        "optimizer" : "RMSprop",
        "learning_rate" : 1e-3
    }


    x = dict()
    obj_img = np.random.random((1,3,28,28))
    img = np.random.random((1,15,28,28))

    x['state'] = Variable(torch.FloatTensor(img))
    x['objective'] = Variable(torch.FloatTensor(obj_img))

    filmed_net = FilmedNet(config)
    filmed_net.forward(x)


