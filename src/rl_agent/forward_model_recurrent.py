import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from rl_agent.film_utils import ResidualBlock, FiLMedResBlock
from .gpu_utils import FloatTensor

class DRQNText(nn.Module):
    def __init__(self, config, n_actions, state_dim, is_multi_objective):
        super(DRQNText, self).__init__()

        # General params
        self.lr = config["learning_rate"]
        self.discount_factor = config["discount_factor"]
        self.is_multi_objective = is_multi_objective
        self.state_dim = state_dim

        # Resblock params
        self.n_regular_block = config["n_regular_block"]
        #self.n_modulated_block = config["n_modulated_block"]
        self.resblock_dropout = config["resblock_dropout"]

        # Head params
        self.n_channel_head = config["head_channel"]
        self.kernel_size_head = config["head_kernel"]
        self.pool_kernel_size_head = config["head_pool_kernel"]

        # If use attention : no head/pooling
        self.use_attention = config['fusing_method_before_recurrent'] == 'attention'
        self.use_film = config["use_film"]

        # Text_encoding params
        self.word_embedding_size = config['word_emb_size']
        self.text_lstm_size = config['text_lstm_size']

        # dimensions are (len_seq, vocab_size) for one sentence
        self.vocab_size = self.state_dim['objective']

        # LSTM Params
        self.n_hidden_lstm = config['n_hidden_lstm_dyn']
        self.lstm_dynamic_num_layer = config['lstm_dyn_n_layer']

        self.input_shape = state_dim['env_state']
        self.objective_shape = state_dim['objective']

        self.n_channel_per_state = self.input_shape[0]

        assert not self.use_film, "Not possible at the moment"
        # if is_multi_objective and self.use_film:
        #     self.film_gen = TextFilmGen(config=config['film_gen_param_text'],
        #                                 n_block_to_modulate=self.n_modulated_block,
        #                                 n_features_per_block=self.n_channel_per_state,
        #                                 input_size=self.lstm_size)

        self.n_actions = n_actions

        self.regular_blocks = nn.ModuleList()
        # self.modulated_blocks = nn.ModuleList()

        # Create resblock, not modulated by FiLM
        for regular_resblock_num in range(self.n_regular_block):
            current_regular_resblock = ResidualBlock(in_dim=self.n_channel_per_state,
                                             with_residual=True)

            self.regular_blocks.append(current_regular_resblock)

        # # Create FiLM-ed resblock
        # for modulated_block_num in range(self.n_modulated_block):
        #     current_modulated_resblock = FiLMedResBlock(in_dim=self.n_channel_per_state,
        #                                       with_residual=True,
        #                                       with_batchnorm=False,
        #                                       with_cond=[True],
        #                                       dropout=self.resblock_dropout)
        #
        #     self.modulated_blocks.append(current_modulated_resblock)

        if self.word_embedding_size > 0:
            self.word_embedding = nn.Embedding(self.vocab_size, self.word_embedding_size)
        else:
            # todo : one hot encoding
            raise NotImplementedError("Word embedding is needed.")

        self.text_objective_lstm = nn.LSTM(input_size=self.word_embedding_size,
                                           hidden_size=self.text_lstm_size,
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


        # The mlp has to deal with the image features AND text features
        # How do you fuse them ? (concatenation, dot-product, attention, don't use)

        # todo : one function "choose_fuse" that return the size of fused and the true fusing fusion
        if config['fusing_method_before_recurrent'] == 'concatenate':
            lstm_input_size = self.text_lstm_size + vizfeat_size_flatten_before_fuse
            self.fuse_before_lstm = self.concatenate_text_vision

        elif config['fusing_method_before_recurrent'] == 'dot':
            self.embedding_size_before_dot = config['embedding_size_before_dot']
            self.visual_embedding_before_dot = nn.Linear(vizfeat_size_flatten_before_fuse, self.embedding_size_before_dot)
            self.text_embedding_before_dot = nn.Linear(self.text_lstm_size, self.embedding_size_before_dot)
            self.fuse_before_lstm = self.dot_product_text_vision

            lstm_input_size = self.embedding_size_before_dot

        elif config['fusing_method_before_recurrent'] == 'attention':
            attention_input_size = self.text_lstm_size + vizfeat_n_feat_map
            hidden_mlp_size = config['hidden_mlp_attention']

            if hidden_mlp_size > 0:
                hidden_layer_att = nn.Linear(attention_input_size, hidden_mlp_size)
                relu = nn.ReLU()
                self.attention_hidden = nn.Sequential(hidden_layer_att, relu)

            else:
                self.attention_hidden = lambda x:x
                hidden_mlp_size = attention_input_size

            self.attention_last = nn.Linear(hidden_mlp_size, 1)

            self.fuse_before_lstm = self.compute_attention
            # text concatenated with vision after visual_attention, so size = lstm_size + width*heigth
            lstm_input_size = self.text_lstm_size + vizfeat_n_feat_map

        elif config['fusing_method_before_recurrent'] == "no_fuse": # Usual Film method, the text is not used

            # if Film no fuse, the size expected by the fc is the size of visual features, flattened
            lstm_input_size = vizfeat_size_flatten_before_fuse
            self.fuse_before_lstm = self.vectorize

        else:
            raise NotImplementedError("Wrong Fusion method : {}, can only be : concatenate, dot, attention or no_fuse, but need to be explicit".format(config['fusing_method_before_recurrent']))


        self.lstm_cells = nn.ModuleList()
        for cell in range(self.lstm_dynamic_num_layer):
            self.lstm_cells.append(nn.LSTMCell(input_size=lstm_input_size, hidden_size= self.n_hidden_lstm))
            # Todo : use layer norm ??

        # Todo : fuse after lstm
        self.final_fc = nn.Linear(in_features=self.n_hidden_lstm, out_features=self.n_actions)

        optimizer = config['optimizer'].lower()


        optim_config = [
            {'params': self.get_all_params_except_film(), 'weight_decay': config['default_w_decay']}, # Default config
        ]

        # if self.use_film:
        #     optim_config.append({'params': self.film_gen.parameters(), 'weight_decay': config['FiLM_decay']})  # Film gen parameters
        #     assert len([i for i in optim_config[1]['params']]) + len([i for i in optim_config[0]['params']]) == len([i for i in self.parameters()])

        if optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(optim_config, lr=self.lr)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(optim_config, lr=self.lr)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(optim_config, lr=self.lr)
        else:
            assert False, 'Optimizer not recognized'

        self.currently_optimizing = False
        self.last_h = Variable(torch.ones(1, self.n_hidden_lstm).type(FloatTensor))
        self.last_c = Variable(torch.ones(1, self.n_hidden_lstm).type(FloatTensor))


    def forward(self, x):
        """
        One timestep forward in time

        2 mode are available :
        current_optimizing = True
        or
        current_optimizing = False

        It only changes WHERE you store h and c (since you do one step of optimizing at every step of an episode
        you want to remember the h used during the episode, and use another h during the optimization
        """

        if self.currently_optimizing:
            h, c = self.optimize_h, self.optimize_c
        else:
            h, c = self.last_h, self.last_c

        x = self.forward_env_features(x)

        for lstm_cell in self.lstm_cells:
            h, c = lstm_cell(x, (h,c))

            if self.currently_optimizing:
                self.optimize_h, self.optimize_c = h, c
            else:
                self.last_h, self.last_c = h, c

        q_values = self.final_fc(h)
        return q_values

    # def forward_multiple_step(self, batch_seq):
    #     """
    #     :param batch_seq: should be of size [Length_seq, Batch, FeatureMaps, Width, Height]
    #     :return: q_values for the last step
    #     """
    #
    #     state_sequence_length = batch_seq.size(0)
    #
    #     h = Variable(torch.ones_like(batch_seq[0]))
    #     c = Variable(torch.ones_like(batch_seq[0]))
    #
    #     # todo : what if there are less step than 'state_sequence_length' ?
    #     # variable number of state_seq_length ?
    #     for step in range(state_sequence_length):
    #         x = batch_seq[step]
    #
    #         x = self.forward_env_features(x)
    #
    #         for lstm_cell in self.lstm_cells:
    #             h, c = lstm_cell(x, (h, c))
    #
    #     q_values = self.final_fc(h)
    #     return q_values


    def forward_env_features(self, x):
        """
        Apply convolution and fusing to raw input x
        :param x:
        :return:
        """

        x, text_objective = x['env_state'], x['objective']

        embedded_objective = self.word_embedding(text_objective)
        _, (text_state, _) = self.text_objective_lstm(embedded_objective)

        # delete the 'sequence' dimension of the lstm, since we take only the last hidden_state
        text_state = text_state.squeeze(0)

        # if self.use_film:
        #     gammas, betas = self.film_gen.forward(text_state)
        # else:
        gammas, betas = None, None

        x = self.compute_conv(x, gammas=gammas, betas=betas)

        # fusing text and images
        x = self.fuse_before_lstm(text=text_state, vision=x)
        return x

    def compute_conv_size(self):

        # Don't convert it to cuda because the model is not yet on GPU (because you're still defining the model here ;)
        tmp = Variable(torch.zeros(1, *self.input_shape))
        return self.compute_conv(tmp).size()

    def compute_conv(self, x, gammas=None, betas=None):

        batch_size = x.size(0)

        # if gammas is None:
        #     # Gammas = all ones
        #     # Betas = all zeros
        #     gammas = Variable(torch.ones(batch_size, self.n_channel_per_state * self.n_modulated_block).type_as(x.data))
        #     betas = Variable(torch.zeros_like(gammas.data).type_as(x.data))

        for i,regular_resblock in enumerate(self.regular_blocks):
            x = regular_resblock.forward(x)

        # for i,modulated_resblock in enumerate(self.modulated_blocks):
        #     gamma_beta_id = slice(self.n_channel_per_state * i, self.n_channel_per_state * (i + 1))
        #     x = modulated_resblock.forward(x, gammas=gammas[:, gamma_beta_id], betas=betas[:, gamma_beta_id])

        if not self.use_attention:
            x = self.head_conv(x)
            x = F.max_pool2d(x, kernel_size=self.pool_kernel_size_head)
        return x

    def optimize_mode(self, optimize, batch_size=None):

        if optimize:
            assert batch_size, "If you want to switch to optimize mode, you have to specify the batch size."
            # todo : ones or zero ??
            self.optimize_h, self.optimize_c = Variable(torch.ones(batch_size, self.n_hidden_lstm).type(FloatTensor)), Variable(torch.ones(batch_size, self.n_hidden_lstm).type(FloatTensor))

        self.currently_optimizing = optimize

    def get_all_params_except_film(self):
        params = []
        for name, param in self.named_parameters():
            if "film_gen" not in name:
                params.append(param)

        return params

    def concatenate_text_vision(self, text, vision):
        vision = vision.view(vision.size(0), -1)
        return torch.cat((vision, text), dim=1)

    def dot_product_text_vision(self, text, vision):
        vision = vision.view(vision.size(0), -1)

        text = self.text_embedding_before_dot(text)
        vision = self.visual_embedding_before_dot(vision)
        return text * vision

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
                current_pixel = vision[:, :, i, j]
                assert current_pixel.dim() == 2
                current_weight = self.attention_last(self.attention_hidden(torch.cat((text, current_pixel), dim=1)))
                attention_weights_list.append(current_weight)

        all_weigths = torch.cat(attention_weights_list, dim=1)
        all_weigths = F.softmax(all_weigths, dim=1).unsqueeze(2)

        vision = vision.view(-1, n_feature_map, height * width)
        vision = torch.bmm(vision, all_weigths)
        vision = vision.squeeze(2)

        return self.concatenate_text_vision(text, vision)

    def vectorize(self, text, vision):
        return vision.view(vision.size(0), -1)


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
