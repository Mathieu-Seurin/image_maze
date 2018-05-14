import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from rl_agent.film_utils import ResidualBlock, FiLMedResBlock, FiLM

from .gpu_utils import FloatTensor

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
        if self.intermediate_kernel_size > 0:
            self.layer1 = nn.Sequential(
                nn.Conv2d(self.n_channel_in, self.n_intermediate_channel, kernel_size=self.intermediate_kernel_size, padding=2),
                nn.BatchNorm2d(self.n_intermediate_channel),
                nn.ReLU(),
                nn.MaxPool2d(2))
        else:
            self.layer1 = lambda x:x
            self.n_intermediate_channel = self.n_channel_in

        if self.final_kernel_size > 0:
            self.layer2 = nn.Sequential(
            nn.Conv2d(self.n_intermediate_channel, self.n_final_channel, kernel_size=self.final_kernel_size, padding=2),
            nn.BatchNorm2d(self.n_final_channel),
            nn.ReLU(),
            nn.MaxPool2d(2))
        else:
            self.layer1 = lambda x: x
            self.n_intermediate_channel = self.n_channel_in

        # Have to determine shape of output before feeding to fc
        tmp = Variable(torch.zeros(1, *self.input_shape))
        conv_out_shape = self.layer2(self.layer1(tmp)).shape
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
        out = self.layer1(x)
        out = self.layer2(out)
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
            self.film_gen = VisionFilmGen(config=config['film_gen_params'],
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
