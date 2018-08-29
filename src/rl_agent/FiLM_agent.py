import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from rl_agent.film_utils import ResidualBlock, FiLMedResBlock, VisionFilmGen

from .gpu_utils import FloatTensor
from .fusing_utils import choose_fusing, lstm_last_step, lstm_whole_seq, TextAttention,\
    ConvPoolReducingLayer, PoolReducingLayer, LinearReducingLayer

class MultiHopTextFilmGen(nn.Module):

    def __init__(self, config, n_block_to_modulate, n_feature_map_per_block, text_size, vision_size=None):
        super(MultiHopTextFilmGen, self).__init__()
        assert vision_size is not None, "For FiLM with feedback loop, need size of visual features"

        self.text_size = text_size
        self.vision_size = vision_size
        self.vision_after_reduce_size_mlp = config["vision_reducing_size_mlp"]
        self.film_gen_hidden_size = config["film_gen_hidden_size"]
        self.use_feedback = config["use_feedback"]

        self.n_feature_map_per_block = n_feature_map_per_block

        self.attention = TextAttention(hidden_mlp_size=config["film_attention_size_hidden"],
                                       text_size=text_size)

        if self.use_feedback:
            if config["vision_reducing_method"] == "mlp" :
                vision_size_flatten = vision_size[1]*vision_size[2]*vision_size[3] # just flatten the input
                self.vision_reducer_layer = LinearReducingLayer(vision_size_flatten=vision_size_flatten,
                                                                output_size=self.vision_after_reduce_size_mlp)
            elif config["vision_reducing_method"] == "conv" :
                self.vision_reducer_layer = ConvPoolReducingLayer(vision_size[1])
            elif config["vision_reducing_method"] == "pool" :
                self.vision_reducer_layer = PoolReducingLayer()
            else:
                raise NotImplementedError("Wrong vision reducing method : {}".format(config["vision_reducing_method"]))

            vision_after_reduce_size = self.compute_reduction_size()[1] # batch_size, n_features
        else:
            vision_after_reduce_size = 0

        self.film_gen_hidden = nn.Linear(self.text_size + vision_after_reduce_size, self.film_gen_hidden_size)
        self.film_gen_last_layer = nn.Linear(self.film_gen_hidden_size, n_feature_map_per_block * 2)
        # for every feature_map, you generate a beta and a gamma, to do : feature_map*gamma + beta
        # So, for every feature_map, 2 parameters are generated


    def forward(self, text, first_layer, vision=None):
        """
        Common interface for all Film Generator
        first_layer indicate that you calling film generator for the first time (needed for init etc...)
        """
        batch_size = text.size(1)

        # if first layer, reset ht to ones only
        if first_layer:
            self.ht = Variable(torch.ones(batch_size, self.text_size).type(FloatTensor))

        # Compute text features
        text_vec = self.attention(text_seq=text, previous_hidden=self.ht)
        # todo layer norm ? not available on 0.3.0
        self.ht = text_vec


        # Compute feedback loop and fuse
        if self.use_feedback:
            vision_feat_reduced = self.vision_reducer_layer(vision)
            film_gen_input = torch.cat((vision_feat_reduced, text_vec), dim=1)
        else:
            film_gen_input = text_vec

        # Generate film parameters
        hidden_film_gen_activ = F.relu(self.film_gen_hidden(film_gen_input))
        gammas_betas = self.film_gen_last_layer(hidden_film_gen_activ)

        return gammas_betas

    def compute_reduction_size(self):

        tmp = Variable(torch.ones(self.vision_size))
        tmp_out = self.vision_reducer_layer(tmp)
        return tmp_out.size()

class SimpleTextFilmGen(nn.Module):

    def __init__(self, config, n_block_to_modulate, n_feature_map_per_block, text_size, vision_size=None):
        super(SimpleTextFilmGen, self).__init__()

        self.n_block_to_modulate = n_block_to_modulate
        self.film_gen_hidden_size = config["film_gen_hidden_size"]

        self.text_size = text_size

        self.n_feature_map_per_block = n_feature_map_per_block
        self.n_features_to_modulate = self.n_block_to_modulate * self.n_feature_map_per_block

        self.film_gen_hidden = nn.Linear(self.text_size, self.film_gen_hidden_size)
        self.film_gen_last_layer = nn.Linear(self.film_gen_hidden_size, n_feature_map_per_block * n_block_to_modulate * 2)
        # for every feature_map, you generate a beta and a gamma, to do : feature_map*gamma + beta
        # So, for every feature_map, 2 parameters are generated

    def forward(self, text, first_layer, vision=None):
        """
        Common interface for all Film Generator
        first_layer indicate that you calling film generator for the first time (needed for init etc...)
        """
        if first_layer:
            self.num_layer_count = 0
            hidden_film_gen_activ = F.relu(self.film_gen_hidden(text))
            self.gammas_betas = self.film_gen_last_layer(hidden_film_gen_activ)

        gamma_beta_id = slice(self.n_feature_map_per_block * self.num_layer_count * 2,
                              self.n_feature_map_per_block * (self.num_layer_count + 1) * 2)

        self.num_layer_count += 1

        return self.gammas_betas[:, gamma_beta_id]

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

        # If use attention as fusing : no head/pooling
        self.use_attention_as_fusing = config['fusing_method'] == 'attention'

        # Film type
        self.use_attention_in_film = False
        self.use_film = config["use_film"]
        if self.use_film:
            self.film_gen_type = config["film_gen_param_text"]["film_type"]
            self.use_attention_in_film = config["film_gen_param_text"]["film_type"] == "multi_hop"

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

        # Lstm encoding text : Do you use all ht or only the last one ?
        if self.use_attention_in_film:
            self.postproc_text = lstm_whole_seq
        else:
            self.postproc_text = lstm_last_step

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
        if self.kernel_size_head != 0 and not self.use_attention_as_fusing:
            self.head_conv = nn.Conv2d(in_channels=self.n_channel_per_state,
                                   out_channels=self.n_channel_head,
                                   kernel_size=self.kernel_size_head)
        else:
            self.head_conv = lambda x:x


        # Select fusing function
        fusing_config = {}
        fusing_config["fusing_method"] = config["fusing_method"]

        vizfeat_shape_before_fuse, intermediate_conv_size = self.compute_conv_size()
        fusing_config["vision_n_feat_map"] = vizfeat_shape_before_fuse[1]

        vizfeat_size_flatten_before_fuse = vizfeat_shape_before_fuse[1]*vizfeat_shape_before_fuse[2]*vizfeat_shape_before_fuse[3]
        fusing_config["vision_size"] = vizfeat_size_flatten_before_fuse

        fusing_config["text_size"] = self.lstm_size
        fusing_config['embedding_size_before_dot'] = config['embedding_size_before_dot']
        fusing_config['hidden_mlp_attention'] = config['hidden_mlp_attention']

        self.fusing_func, fc_input_size = choose_fusing(fusing_config)

        self.fc1 = nn.Linear(in_features=fc_input_size, out_features=self.n_hidden)
        self.fc2 = nn.Linear(in_features=self.n_hidden, out_features=self.n_actions)


        # If Film
        if is_multi_objective and self.use_film:

            if self.film_gen_type == "simple":
                self.film_gen = SimpleTextFilmGen(config=config['film_gen_param_text'],
                                                  n_block_to_modulate=self.n_modulated_block,
                                                  n_feature_map_per_block=self.n_channel_per_state,
                                                  text_size=self.lstm_size)

            elif self.film_gen_type == "multi_hop":
                self.film_gen = MultiHopTextFilmGen(config=config['film_gen_param_text'],
                                                    n_block_to_modulate=self.n_modulated_block,
                                                    n_feature_map_per_block=self.n_channel_per_state,
                                                    text_size=self.lstm_size,
                                                    vision_size=intermediate_conv_size)
            else:
                raise NotImplementedError("Wrong Film generator type : given '{}'".format(self.film_gen_type))

        optimizer = config['optimizer'].lower()


        optim_config = [
            {'params': self.get_all_params_except_film(), 'weight_decay': config['default_w_decay']}, # Default config
        ]

        if self.use_film:
            optim_config.append({'params': self.film_gen.parameters(), 'weight_decay': config['FiLM_decay']})  # Film gen parameters
            assert len([i for i in optim_config[1]['params']]) + len([i for i in optim_config[0]['params']]) == len([i for i in self.parameters()])

        if optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(optim_config, lr=self.lr)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(optim_config, lr=self.lr)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(optim_config, lr=self.lr)
        else:
            assert False, 'Optimizer not recognized'

    def forward(self, x):

        x, text_objective = x['env_state'], x['objective']

        embedded_objective = self.word_embedding(text_objective)
        all_ht, (last_ht, _) = self.lstm(embedded_objective)

        text_state = self.postproc_text(all_ht=all_ht, last_ht=last_ht)

        x = self.compute_conv(x, text_state, still_building_model=False)

        # fusing text and images
        x = self.fusing_func(text=text_state, vision=x)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.fc_dropout, training=self.training)
        x = self.fc2(x)
        return x

    def compute_conv_size(self):

        # Don't convert it to cuda because the model is not yet on GPU (because you're still defining the model here ;)
        tmp = Variable(torch.zeros(1, *self.input_shape))
        return self.compute_conv(tmp, still_building_model=True).size(), self.intermediate_conv_size

    def compute_conv(self, x, text_state=None, still_building_model=False):
        """
        :param x: vision input with batch dimension first
        :param text_state: all hidden states of the lstm encoder
        :param still_building_model: needed if you use this function just to get the output size
        :return: return visual features, modulated if FiLM
        """

        if self.use_film:
            if not still_building_model:
                assert text_state is not None, "if you use film, need to provide text as input too"

        batch_size = x.size(0)

        # Regular resblock, easy
        for i,regular_resblock in enumerate(self.regular_blocks):
            x = regular_resblock.forward(x)
            self.intermediate_conv_size = x.size()


        #Modulated block : first compute FiLM weights, then send them to the resblock layers
        for i,modulated_resblock in enumerate(self.modulated_blocks):

            if still_building_model or not self.use_film :
                # Gammas = all ones   Betas = all zeros
                gammas = Variable(torch.ones(batch_size, self.n_channel_per_state).type_as(x.data))
                betas = Variable(torch.zeros_like(gammas.data).type_as(x.data))
            else: # use film
                gammas_betas = self.film_gen.forward(text_state, first_layer= i==0, vision=x)
                assert gammas_betas.size(1)%2 == 0, "Problem, more gammas than betas (or vice versa)"
                middle = gammas_betas.size(1)//2
                gammas = gammas_betas[:,:middle]
                betas = gammas_betas[:, middle:]

            x = modulated_resblock.forward(x, gammas=gammas, betas=betas)
            self.intermediate_conv_size = x.size()

        if not self.use_attention_as_fusing:
            x = self.head_conv(x)
            x = F.max_pool2d(x, kernel_size=self.pool_kernel_size_head)

        return x

    def get_all_params_except_film(self):

        params = []
        for name, param in self.named_parameters():
            if "film_gen" not in name:
                params.append(param)

        return params



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
