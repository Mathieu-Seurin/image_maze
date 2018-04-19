import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import logging

# GPU compatibility setup
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class DQN(nn.Module):

    def __init__(self, config, n_out):
        super(DQN, self).__init__()
        self.output_size = n_out

        conv_layers = nn.ModuleList()
        dense_layers = nn.ModuleList()

        self.input_resolution = config['input_resolution']
        self.n_channels = config['n_channels']
        self.conv_shapes = config['conv_shapes']
        self.dense_shapes = config['dense_shapes'] + [self.output_size]
        self.use_batch_norm = config['use_batch_norm'] == 'True'
        self.lr = config['learning_rate']
        self.gamma = config['gamma']

        # At least 1 conv, then dense head
        for idx, shape in enumerate(self.conv_shapes):
            if idx == 0:
                conv_layers.append(nn.Conv2d(self.n_channels, shape, kernel_size=3, stride=2))
            else:
                conv_layers.append(nn.Conv2d(tmp, shape, kernel_size=5, stride=2))
            conv_layers.append(nn.ReLU())
            if self.use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(shape))
            tmp = shape
        self.conv_layers = conv_layers

        # Infer shape after flattening
        tmp = self._get_conv_output_size([self.n_channels,] + self.input_resolution)

        for idx, shape in enumerate(self.dense_shapes):
            dense_layers.append(nn.Linear(tmp, shape))
            if idx < len(self.dense_shapes)-1:
                dense_layers.append(nn.ReLU())
                if self.use_batch_norm:
                    dense_layers.append(nn.BatchNorm1d(shape))
            tmp = shape
        self.dense_layers = dense_layers

        logging.info('Model summary :')
        for l in self.conv_layers:
            logging.info(l)
        for l in self.dense_layers:
            logging.info(l)

        if config['optimizer'] == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        elif config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            assert False, 'Optimizer not recognized'


    def _get_conv_output_size(self, shape):
        bs = 1
        inpt = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_conv(inpt)
        total_size = output_feat.data.view(bs, -1).size(1)
        return total_size

    def _forward_conv(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def _forward_dense(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        return self._forward_dense(x)


class SoftmaxDQN(nn.Module):

    def __init__(self, config, n_out):
        super(SoftmaxDQN, self).__init__()
        self.output_size = n_out

        conv_layers = nn.ModuleList()
        dense_layers = nn.ModuleList()

        self.input_resolution = config['input_resolution']
        self.n_channels = config['n_channels']
        self.conv_shapes = config['conv_shapes']
        self.dense_shapes = config['dense_shapes'] + [self.output_size]
        self.use_batch_norm = config['use_batch_norm'] == 'True'
        self.lr = config['learning_rate']

        # At least 1 conv, then dense head
        for idx, shape in enumerate(self.conv_shapes):
            if idx == 0:
                conv_layers.append(nn.Conv2d(self.n_channels, shape, kernel_size=3, stride=2))
            else:
                conv_layers.append(nn.Conv2d(tmp, shape, kernel_size=5, stride=2))
            conv_layers.append(nn.ReLU())
            if self.use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(shape))
            tmp = shape
        self.conv_layers = conv_layers

        # Infer shape after flattening
        tmp = self._get_conv_output_size([self.n_channels,] + self.input_resolution)

        for idx, shape in enumerate(self.dense_shapes):
            dense_layers.append(nn.Linear(tmp, shape))
            if idx < len(self.dense_shapes)-1:
                dense_layers.append(nn.ReLU())
                if self.use_batch_norm:
                    # dense_layers.append(nn.BatchNorm1d(shape))
                    print('BatchNorm in dense layers not working')
            else:
                dense_layers.append(nn.Softmax(dim=1))
            tmp = shape
        self.dense_layers = dense_layers

        logging.info('Model summary :')
        for l in self.conv_layers:
            logging.info(l)
        for l in self.dense_layers:
            logging.info(l)

        if config['optimizer'] == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        elif config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            assert False, 'Optimizer not recognized'


    def _get_conv_output_size(self, shape):
        bs = 1
        inpt = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_conv(inpt)
        total_size = output_feat.data.view(bs, -1).size(1)
        return total_size

    def _forward_conv(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def _forward_dense(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        return self._forward_dense(x)
