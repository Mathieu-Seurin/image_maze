import torch
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


################################################
#
#       OBSOLETE VISION OBJECTIVE MODULE
#
################################################

class VisionFilmGen(nn.Module):

    def __init__(self, config, n_block_to_modulate, n_features_per_block, input_shape):
        super(VisionFilmGen, self).__init__()

        self.n_block_to_modulate = n_block_to_modulate

        self.input_shape = input_shape
        self.n_channel_in = self.input_shape[0]

        self.n_features_per_block = n_features_per_block
        self.n_features_to_modulate = self.n_block_to_modulate*self.n_features_per_block

        self.n_intermediate_channel = config['n_intermediate_channel']
        self.intermediate_kernel_size = config['intermediate_kernel_size']

        self.n_final_channel = config['n_final_channel']
        self.final_kernel_size = config['final_kernel_size']

        self.n_hidden_gamma = config['n_hidden_gamma']
        self.n_hidden_beta = config['n_hidden_beta']

        self.dropout = config['dropout']

        # Convolution for objective as an image
        self.conv_layers = nn.ModuleList()
        if self.intermediate_kernel_size > 0:

            self.conv_layers.append(nn.Conv2d(self.n_channel_in, self.n_intermediate_channel, kernel_size=self.intermediate_kernel_size, padding=2))
            self.conv_layers.append(nn.BatchNorm2d(self.n_intermediate_channel))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2))
        else:
            self.n_intermediate_channel = self.n_channel_in

        if self.final_kernel_size > 0:

            self.conv_layers.append(nn.Conv2d(self.n_intermediate_channel, self.n_final_channel, kernel_size=self.final_kernel_size, padding=2))
            self.conv_layers.append(nn.BatchNorm2d(self.n_final_channel))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2))

        else:
            self.n_intermediate_channel = self.n_channel_in

        # Have to determine shape of output before feeding to fc
        tmp = Variable(torch.zeros(1, *self.input_shape))
        for module in self.conv_layers :
            tmp = module(tmp)
        conv_out_shape = tmp.shape
        n_features_after_conv = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]

        # Add hidden layer (and dropout) before computing gammas and betas
        if self.n_hidden_gamma > 0:
            hidden_layer_gamma = nn.Linear(n_features_after_conv, self.n_hidden_gamma)
            dropout_gamma = nn.Dropout(self.dropout)
            relu = nn.ReLU()
            self.hidden_layer_gamma = nn.Sequential(hidden_layer_gamma, dropout_gamma, relu)
        else:
            self.hidden_layer_gamma = lambda x:x # Identity
            self.n_hidden_gamma = n_features_after_conv

        if self.n_hidden_beta > 0:
            hidden_layer_beta = nn.Linear(n_features_after_conv, self.n_hidden_beta)
            dropout_beta = nn.Dropout(self.dropout)
            relu = nn.ReLU()
            self.hidden_layer_beta = nn.Sequential(hidden_layer_beta, dropout_beta, relu)

        else:
            self.hidden_layer_beta = lambda x:x # Identity
            self.n_hidden_beta = n_features_after_conv

        # compute gammas and betas
        self.fc_gammas = nn.Linear(self.n_hidden_gamma, self.n_features_to_modulate)
        self.fc_betas = nn.Linear(self.n_hidden_beta, self.n_features_to_modulate)

    def forward(self, x):
        # out = self.layer1(x)
        # out = self.layer2(out)

        out = x
        for module in self.conv_layers :
            out = module(out)

        out.contiguous()
        out = out.view(out.size(0), -1)

        gammas = self.hidden_layer_gamma(out)
        betas = self.hidden_layer_beta(out)

        gammas = self.fc_gammas(gammas)
        betas = self.fc_betas(betas)
        return gammas, betas


class FilmedNet(nn.Module):
    def __init__(self, config, n_actions, state_dim, is_multi_objective):
        super(FilmedNet, self).__init__()

        # General params
        self.lr = config["learning_rate"]
        self.discount_factor = config["discount_factor"]
        self.use_film = config["use_film"]
        self.is_multi_objective = is_multi_objective

        # Resblock params
        self.n_regular_block = config["n_regular_block"]
        self.n_modulated_block = config["n_modulated_block"]
        self.resblock_dropout = config["resblock_dropout"]

        # Head params
        self.n_channel_head = config["head_channel"]
        self.kernel_size_head = config["head_kernel"]
        self.pool_kernel_size_head = config["head_pool_kernel"]

        # FC params
        self.n_hidden = config['n_hidden']
        self.fc_dropout = config["fc_dropout"]


        if self.is_multi_objective and not self.use_film:
            self.input_shape = state_dim['concatenated']
        else:
            self.input_shape = state_dim['env_state']
            self.objective_shape = state_dim['objective']

        self.n_channel_per_state = self.input_shape[0]

        if is_multi_objective and self.use_film:
            self.film_gen = VisionFilmGen(config=config['film_gen_param_vision'],
                                          n_block_to_modulate=self.n_modulated_block,
                                          n_features_per_block=self.n_channel_per_state,
                                          input_shape=self.objective_shape)


        self.n_actions = n_actions

        self.regular_blocks = nn.ModuleList()
        self.modulated_blocks = nn.ModuleList()

        # Create resblock, not modulated by FiLM
        for regular_resblock_num in range(self.n_regular_block):
            current_regular_resblock = ResidualBlock(in_dim=self.n_channel_per_state,
                                             with_residual=True)

            self.regular_blocks.append(current_regular_resblock)

        # Create FiLM-ed resblock
        for modulated_block_num in range(self.n_modulated_block):
            current_modulated_resblock = FiLMedResBlock(in_dim=self.n_channel_per_state,
                                              with_residual=True,
                                              with_batchnorm=False,
                                              with_cond=[True],
                                              dropout=self.resblock_dropout)

            self.modulated_blocks.append(current_modulated_resblock)


        # head
        self.head_conv = nn.Conv2d(in_channels=self.n_channel_per_state,
                                   out_channels=self.n_channel_head,
                                   kernel_size=self.kernel_size_head)

        in_features = self.conv_output_size()

        self.fc1 = nn.Linear(in_features=in_features, out_features=self.n_hidden)
        self.fc2 = nn.Linear(in_features=self.n_hidden, out_features=self.n_actions)

        optimizer = config['optimizer'].lower()
        if optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
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
            gammas = Variable(torch.ones(batch_size, self.n_channel_per_state * self.n_modulated_block).type(FloatTensor))
            betas = Variable(torch.zeros_like(gammas.data).type(FloatTensor))

        for i,regular_resblock in enumerate(self.regular_blocks):
            x = regular_resblock.forward(x)

        for i,modulated_resblock in enumerate(self.modulated_blocks):
            gamma_beta_id = slice(self.n_channel_per_state * i, self.n_channel_per_state * (i + 1))
            x = modulated_resblock.forward(x, gammas=gammas[:, gamma_beta_id], betas=betas[:, gamma_beta_id])

        x = self.head_conv(x)

        x = F.max_pool2d(x, kernel_size=self.pool_kernel_size_head)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.fc_dropout, training=self.training)
        x = self.fc2(x)
        return x

    def conv_output_size(self):
        # Have to determine shape of output before feeding to fc
        gammas = Variable(torch.ones(1, self.n_channel_per_state * self.n_modulated_block))
        betas = Variable(torch.zeros_like(gammas.data))

        tmp = Variable(torch.zeros(1, *self.input_shape))
        for i,regular_resblock in enumerate(self.regular_blocks):
            tmp = regular_resblock.forward(tmp)
        for i,modulated_resblock in enumerate(self.modulated_blocks):
            gamma_beta_id = slice(self.n_channel_per_state * i, self.n_channel_per_state * (i + 1))
            tmp = modulated_resblock.forward(tmp, gammas=gammas[:, gamma_beta_id], betas=betas[:, gamma_beta_id])

        tmp = self.head_conv(tmp)
        tmp = F.max_pool2d(tmp, kernel_size=self.pool_kernel_size_head)

        conv_out_shape = tmp.shape

        return conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]