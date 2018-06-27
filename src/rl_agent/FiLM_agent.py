import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from rl_agent.film_utils import ResidualBlock, FiLMedResBlock

from .gpu_utils import FloatTensor
from torch.autograd import Variable

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


class TextFilmGen(nn.Module):

    def __init__(self, config, n_block_to_modulate, n_features_per_block, input_size):
        super(TextFilmGen, self).__init__()

        self.n_block_to_modulate = n_block_to_modulate

        self.input_size = input_size

        self.n_features_per_block = n_features_per_block
        self.n_features_to_modulate = self.n_block_to_modulate * self.n_features_per_block

        self.n_hidden_gamma = config['n_hidden_gamma']
        self.n_hidden_beta = config['n_hidden_beta']

        self.dropout = config['dropout']

        # todo : one common layer ??
        # Add hidden layer (and dropout) before computing gammas and betas
        if self.n_hidden_gamma > 0:
            hidden_layer_gamma = nn.Linear(input_size, self.n_hidden_gamma)
            dropout_gamma = nn.Dropout(self.dropout)
            relu = nn.ReLU()
            self.hidden_layer_gamma = nn.Sequential(hidden_layer_gamma, dropout_gamma, relu)
        else:
            self.hidden_layer_gamma = lambda x: x  # Identity
            self.n_hidden_gamma = input_size

        if self.n_hidden_beta > 0:
            hidden_layer_beta = nn.Linear(input_size, self.n_hidden_beta)
            dropout_beta = nn.Dropout(self.dropout)
            relu = nn.ReLU()
            self.hidden_layer_beta = nn.Sequential(hidden_layer_beta, dropout_beta, relu)

        else:
            self.hidden_layer_beta = lambda x: x  # Identity
            self.n_hidden_beta = input_size

        if config['common_layer'] is True:
            self.n_hidden_beta = self.n_hidden_gamma
            self.hidden_layer_beta = self.hidden_layer_gamma

        # compute gammas and betas
        self.fc_gammas = nn.Linear(self.n_hidden_gamma, self.n_features_to_modulate)
        self.fc_betas = nn.Linear(self.n_hidden_beta, self.n_features_to_modulate)

    def forward(self, x):

        gammas = self.fc_gammas(self.hidden_layer_gamma(x))
        betas = self.fc_betas(self.hidden_layer_beta(x))
        return gammas, betas

class FilmedNetText(nn.Module):
    def __init__(self, config, n_actions, state_dim, is_multi_objective):
        super(FilmedNetText, self).__init__()

        # General params
        self.lr = config["learning_rate"]
        self.discount_factor = config["discount_factor"]
        self.is_multi_objective = is_multi_objective
        self.state_dim = state_dim

        # Resblock params
        self.n_regular_block = config["n_regular_block"]
        self.n_modulated_block = config["n_modulated_block"]
        self.resblock_dropout = config["resblock_dropout"]

        # Head params
        self.n_channel_head = config["head_channel"]
        self.kernel_size_head = config["head_kernel"]
        self.pool_kernel_size_head = config["head_pool_kernel"]

        # If use attention : no head/pooling
        self.use_attention = config['fusing_method'] == 'attention'
        self.use_film = config["fusing_method"] == "film"

        # Text_encoding params
        self.word_embedding_size = config['word_emb_size']
        self.lstm_size = config['lstm_size']

        # dimensions are (len_seq, vocab_size) for one sentence
        self.vocab_size = self.state_dim['objective']

        # FC params
        self.n_hidden = config['n_hidden']
        self.fc_dropout = config["fc_dropout"]

        self.input_shape = state_dim['env_state']
        self.objective_shape = state_dim['objective']

        self.n_channel_per_state = self.input_shape[0]

        if is_multi_objective and self.use_film:
            self.film_gen = TextFilmGen(config=config['film_gen_param_text'],
                                        n_block_to_modulate=self.n_modulated_block,
                                        n_features_per_block=self.n_channel_per_state,
                                        input_size=self.lstm_size)

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

        if self.word_embedding_size > 0:
            self.word_embedding = nn.Embedding(self.vocab_size, self.word_embedding_size)
        else:
            # todo : one hot encoding
            raise NotImplementedError("Word embedding is needed.")

        self.lstm = nn.LSTM(input_size=self.word_embedding_size,
                            hidden_size=self.lstm_size,
                            num_layers=1,
                            batch_first=True)


        # head
        if self.kernel_size_head != 0 and not self.use_attention:
            self.head_conv = nn.Conv2d(in_channels=self.n_channel_per_state,
                                   out_channels=self.n_channel_head,
                                   kernel_size=self.kernel_size_head)
        else:
            self.head_conv = lambda x:x

        vizfeat_shape_before_fuse = self.compute_conv_size()
        vizfeat_n_feat_map = vizfeat_shape_before_fuse[1]
        vizfeat_height = vizfeat_shape_before_fuse[2]
        vizfeat_width = vizfeat_shape_before_fuse[3]

        vizfeat_size_flatten_before_fuse = vizfeat_n_feat_map*vizfeat_height*vizfeat_width

        # if Film, the size expected by the fc is the size of visual features, flattened
        fc_input_size = vizfeat_size_flatten_before_fuse

        # If you don't use film, the mlp has to deal with the image features AND text features
        # How do you fuse them ? (concatenation, dot-product, attention)
        if not self.use_film:

            if config['fusing_method'] == 'concatenate':
                fc_input_size = self.lstm_size + vizfeat_size_flatten_before_fuse
                self.fuse = self.concatenate_text_vision

            elif config['fusing_method'] == 'dot':
                self.embedding_size_before_dot = config['embedding_size_before_dot']
                self.visual_embedding_before_dot = nn.Linear(vizfeat_size_flatten_before_fuse, self.embedding_size_before_dot)
                self.text_embedding_before_dot = nn.Linear(self.lstm_size, self.embedding_size_before_dot)
                self.fuse = self.dot_product_text_vision

                fc_input_size = self.embedding_size_before_dot

            elif config['fusing_method'] == 'attention':
                attention_input_size = self.lstm_size + vizfeat_n_feat_map
                hidden_mlp_size = config['hidden_mlp_attention']

                if hidden_mlp_size > 0:
                    hidden_layer_att = nn.Linear(attention_input_size, hidden_mlp_size)
                    relu = nn.ReLU()
                    self.attention_hidden = nn.Sequential(hidden_layer_att, relu)

                else:
                    self.attention_hidden = lambda x:x
                    hidden_mlp_size = attention_input_size

                self.attention_last = nn.Linear(hidden_mlp_size, 1)

                self.fuse = self.compute_attention
                # text concatenated with vision after visual_attention, so size = lstm_size + width*heigth
                fc_input_size = self.lstm_size + vizfeat_n_feat_map
            else:
                raise NotImplementedError("Wrong Fusion method : {}, can only be : concatenate, dot, attention ".format(config['fusing_method']))

        self.fc1 = nn.Linear(in_features=fc_input_size, out_features=self.n_hidden)
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

        x, text_objective = x['env_state'], x['objective']

        embedded_objective = self.word_embedding(text_objective)
        _, (text_state, _) = self.lstm(embedded_objective)

        #delete the 'sequence' dimension of the lstm, since we take only the last hidden_state
        text_state = text_state.squeeze(0)

        if self.use_film:
            gammas, betas = self.film_gen.forward(text_state)
        else:
            gammas, betas = None, None

        x = self.compute_conv(x, gammas=gammas, betas=betas)

        # fusing text and images
        if not self.use_film:
            x = self.fuse(text=text_state, vision=x)
        else:
            x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.fc_dropout, training=self.training)
        x = self.fc2(x)
        return x

    def compute_conv_size(self):

        # Don't convert it to cuda because the model is not yet on GPU (because you're still defining the model here ;)
        tmp = Variable(torch.zeros(1, *self.input_shape))
        return self.compute_conv(tmp).size()

    def compute_conv(self, x, gammas=None, betas=None):

        batch_size = x.size(0)

        if gammas is None:
            # Gammas = all ones
            # Betas = all zeros
            gammas = Variable(torch.ones(batch_size, self.n_channel_per_state * self.n_modulated_block).type_as(x.data))
            betas = Variable(torch.zeros_like(gammas.data).type_as(x.data))

        for i,regular_resblock in enumerate(self.regular_blocks):
            x = regular_resblock.forward(x)
        for i,modulated_resblock in enumerate(self.modulated_blocks):
            gamma_beta_id = slice(self.n_channel_per_state * i, self.n_channel_per_state * (i + 1))
            x = modulated_resblock.forward(x, gammas=gammas[:, gamma_beta_id], betas=betas[:, gamma_beta_id])

        if not self.use_attention:
            x = self.head_conv(x)
            x = F.max_pool2d(x, kernel_size=self.pool_kernel_size_head)

        return x

    def concatenate_text_vision(self, text, vision):
        vision = vision.view(vision.size(0), -1)
        return torch.cat((vision, text), dim=1)

    def dot_product_text_vision(self, text, vision):
        vision = vision.view(vision.size(0), -1)

        text = self.text_embedding_before_dot(text)
        vision = self.visual_embedding_before_dot(vision)
        return text*vision

    def compute_attention(self, text, vision):
        """
        :param text: lstm-encoded text. dim is (batch, hidden_lstm_size)
        :param vision: cnn-encoded image. dim is (batch, n_feature_map, width, height)
        :return: vision after visual attention is applied. dim is (batch, n_feature_map)
        """
        n_feature_map = vision.size(1)
        width = vision.size(2)
        height = vision.size(3)

        attention_weights_list = []
        # compute attention for every pixel, compute the sum
        for i in range(width):
            for j in range(height):
                current_pixel = vision[:,:,i,j]
                assert current_pixel.dim() == 2
                current_weight = self.attention_last(self.attention_hidden(torch.cat((text, current_pixel), dim=1)))
                attention_weights_list.append(current_weight)

        all_weigths = torch.cat(attention_weights_list, dim=1)
        all_weigths = F.softmax(all_weigths, dim=1).unsqueeze(2)

        vision = vision.view(-1,n_feature_map,height*width)
        vision = torch.bmm(vision, all_weigths)
        vision = vision.squeeze(2)

        return self.concatenate_text_vision(text, vision)



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
