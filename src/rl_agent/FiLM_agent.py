import math
import ipdb as pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_agent.film_utils import init_modules

class FiLM_agent(nn.Module):

    def forward(self, x):
        pass


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

    if self.with_input_proj % 2 == 0:
      raise(NotImplementedError)
    if self.kernel_size % 2 == 0:
      raise(NotImplementedError)
    if self.num_layers >= 2:
      raise(NotImplementedError)

    if self.condition_method == 'block-input-film' and self.with_cond[0]:
      self.film = FiLM()
    if self.with_input_proj:
      self.input_proj = nn.Conv2d(in_dim + (num_extra_channels if self.extra_channel_freq >= 1 else 0),
                                  in_dim, kernel_size=self.with_input_proj, padding=self.with_input_proj // 2)

    self.conv1 = nn.Conv2d(in_dim + self.num_cond_maps +
                           (num_extra_channels if self.extra_channel_freq >= 2 else 0),
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
    if self.debug_every <= -2:
      pdb.set_trace()

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