import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, kaiming_uniform

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=False):
        if out_dim is None:
            out_dim = in_dim
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_dim)
            self.bn2 = nn.BatchNorm2d(out_dim)
        self.with_residual = with_residual
        
        if in_dim == out_dim or not with_residual:
            self.proj = None
        else:
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.with_batchnorm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = self.conv2(F.relu(self.conv1(x)))
        res = x if self.proj is None else self.proj(x)
        if self.with_residual:
            out = F.relu(res + out)
        else:
            out = F.relu(out)
        return out


def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_params(m.weight)

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
                 with_cond=[False], dropout=0, n_extra_channels=0, extra_channel_freq=1,
                 with_input_proj=0, n_cond_maps=0, kernel_size=3, batchnorm_affine=False,
                 n_layers=1, condition_method='bn-film', debug_every=float('inf')):

        if out_dim is None:
            out_dim = in_dim

        super(FiLMedResBlock, self).__init__()
        self.with_residual = with_residual
        self.with_batchnorm = with_batchnorm
        self.with_cond = with_cond
        self.dropout = dropout
        self.extra_channel_freq = 0 if n_extra_channels == 0 else extra_channel_freq
        self.with_input_proj = with_input_proj  # Kernel size of input projection
        self.n_cond_maps = n_cond_maps
        self.kernel_size = kernel_size
        self.batchnorm_affine = batchnorm_affine
        self.n_layers = n_layers
        self.condition_method = condition_method
        self.debug_every = debug_every

        # if self.with_input_proj % 2 == 0:
        #   raise(NotImplementedError)
        if self.kernel_size % 2 == 0:
            raise(NotImplementedError)
        if self.n_layers >= 2:
            raise(NotImplementedError)

        if self.condition_method == 'block-input-film' and self.with_cond[0]:
            self.film = FiLM()
        if self.with_input_proj:
            self.input_proj = nn.Conv2d(in_dim + (n_extra_channels if self.extra_channel_freq >= 1 else 0),
                                      in_dim, kernel_size=self.with_input_proj, padding=self.with_input_proj // 2)

        n_extra_channels = n_extra_channels if self.extra_channel_freq >= 2 else 0
        self.conv1 = nn.Conv2d(in_dim + self.n_cond_maps + n_extra_channels,
                                out_dim, kernel_size=self.kernel_size,
                                padding=self.kernel_size // 2)

        if self.condition_method == 'conv-film' and self.with_cond[0]:
            self.film = FiLM()
        if self.with_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_dim, affine=((not self.with_cond[0]) or self.batchnorm_affine))
        if self.condition_method == 'bn-film' and self.with_cond[0]:
            self.film = FiLM()
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
            out = F.dropout2d(out, training=self.training)
        out = F.relu(out)
        if self.condition_method == 'relu-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)

        # ResBlock remainder
        if self.with_residual:
            out = x + out
        if self.condition_method == 'block-output-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)

        return out