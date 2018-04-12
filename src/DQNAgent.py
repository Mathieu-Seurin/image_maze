"""
Gym complient implementation of pytorch-based agents
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import json
import random
import math
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
from skvideo.io import FFmpegWriter as VideoWriter
from utils import ReplayMemory, write_json_config_file, get_screen, Transition
from utils import Flatten, check_params_changed, freeze_as_np_dict
import logging


# GPU compatibility setup
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class DQN(nn.Module):

    def __init__(self, filename):
        super(DQN, self).__init__()
        self.filename = filename
        conv_layers = nn.ModuleList()
        dense_layers = nn.ModuleList()

        with open(filename, 'r') as f:
            params = json.load(f)

        self.input_resolution = params['input_resolution']
        self.n_channels = params['n_channels']
        self.output_size = params['n_out']
        self.batch_size = params['batch_size']


        # At least 1 conv, then dense head
        for idx, shape in enumerate(params['conv_shapes']):
            if idx == 0:
                conv_layers.append(nn.Conv2d(self.n_channels, shape, kernel_size=3, stride=2))
            else:
                conv_layers.append(nn.Conv2d(tmp, shape, kernel_size=5, stride=2))
            conv_layers.append(nn.ReLU())
            if params['use_batch_norm']:
                conv_layers.append(nn.BatchNorm2d(shape))
            tmp = shape
        self.conv_layers = conv_layers

        # Infer shape after flattening
        tmp = self._get_conv_output_size([self.n_channels,] + self.input_resolution)

        for idx, shape in enumerate(params['dense_shapes'] + [self.output_size]):
            dense_layers.append(nn.Linear(tmp, shape))
            if idx < len(params['dense_shapes']):
                dense_layers.append(nn.ReLU())
                if params['use_batch_norm']:
                    dense_layers.append(nn.BatchNorm2d(shape))
            tmp = shape
        self.dense_layers = dense_layers

        logging.info('Model summary :')
        for l in self.conv_layers:
            logging.info(l)
        for l in self.dense_layers:
            logging.info(l)


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



class DQNagent(object):
    def __init__(self, filename='dqn0'):
        self.filename = './trained_agents/'+filename
        self.policy_net = DQN(self.filename + '.cfg')
        self.target_net = DQN(self.filename + '.cfg')
        self.memory = ReplayMemory(16384)
        self.gamma = 0.999

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            idx = LongTensor([[random.randrange(self.policy_net.output_size)]])
        else:
            idx = self.policy_net(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        return idx

    def update(self, batch_size=16):
        if len(self.memory.memory) < batch_size:
            batch_size = len(self.memory.memory)

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))


        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        expected_state_action_values = Variable(expected_state_action_values.data)

        loss = F.mse_loss(state_action_values, expected_state_action_values)


        old_params = freeze_as_np_dict(self.policy_net.state_dict())
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            logging.debug(param.grad.data.sum())
            param.grad.data.clamp_(-1., 1.)
        self.optimizer.step()

        new_params = freeze_as_np_dict(self.policy_net.state_dict())
        check_params_changed(old_params, new_params)
        return loss.data[0]


    def train(self, env, n_epochs=30, epsilon_init=1., epsilon_schedule='exp', eps_decay=None, lr=0.001, batch_size=32):
        if epsilon_schedule == 'linear':
            eps_range = np.linspace(epsilon_init, 0., n_epochs)
        elif epsilon_schedule=='constant':
            eps_range = [epsilon_init for _ in range(n_epochs)]
        elif epsilon_schedule=='exp':
            if not eps_decay:
                eps_decay = n_epochs // 4
            eps_range = [epsilon_init * math.exp(-1. * i / eps_decay) for i in range(n_epochs)]

        history_file = open(self.filename + 'history', mode='a+')
        self.policy_net = self.policy_net.cuda()
        self.target_net = self.target_net.cuda()
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)

        losses, rewards, change_history = [], [], []

        for epoch in range(n_epochs):
            env.reset()
            last_screen = get_screen(env)
            current_screen = get_screen(env)
            state = current_screen - last_screen
            done = False
            epoch_losses = []
            epoch_rewards = []
            video = []

            while not done:
                if epoch % 10 == 1:
                    video.append(last_screen)
                action = self.select_action(state, eps_range[epoch])

                _, reward, done, _ = env.step(action[0, 0])

                last_screen = current_screen
                current_screen = get_screen(env)

                reward = Tensor([reward])
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                self.memory.push(state, action, next_state, reward)
                state = next_state
                loss = self.update(batch_size=batch_size)

                epoch_losses.append(loss)
                epoch_rewards.append(reward)

            history_file.write('Epoch {}: loss= {}, reward= {}, duration= {}\n'.format(
                epoch, np.mean(epoch_losses), np.sum(epoch_rewards), len(epoch_rewards)))

            logging.info('Epoch {}: loss= {}, reward= {}, duration= {}'.format(
                epoch, np.mean(epoch_losses), np.sum(epoch_rewards), len(epoch_rewards)))
            losses.append(np.mean(epoch_losses))
            rewards.append(np.sum(epoch_rewards))


            if epoch % 10 == 1:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.save(ext=str(epoch))
                self.make_video(video, ext='_train_' + str(epoch))

                with open(self.filename+'.train_losses', 'a+') as f:
                    for l in losses:
                        f.write(str(l)+'\n')
                losses = []
                with open(self.filename+'.train_rewards', 'a+') as f:
                    for r in rewards:
                        f.write(str(r)+'\n')
                rewards = []
        self.save()

    def test(self, env,  n_epochs=30, verbose=False):
        rewards  = []
        self.policy_net = self.policy_net.cuda()
        self.target_net = self.target_net.cuda()
        self.target_net.eval()

        for epoch in range(n_epochs):
            env.reset()
            done = False
            epoch_rewards = []
            video = []

            last_screen = get_screen(env)
            current_screen = get_screen(env)
            state = current_screen - last_screen

            while not done:
                if epoch % 5 == 0:
                    video.append(last_screen)
                action = self.select_action(state, 0.)

                _, reward, done, _ = env.step(action[0, 0])
                last_screen = current_screen
                current_screen = get_screen(env)

                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                epoch_rewards.append(reward)
                reward = Tensor([reward])
                state = next_state

                logging.debug('Test epoch {} :  reward= {}, duration= {}'.format(epoch,
                     np.sum(epoch_rewards), len(epoch_rewards)))
            rewards.append(np.sum(epoch_rewards))

            if epoch % 5 == 0:
                self.make_video(video, ext='_test_' + str(epoch))

            logging.info('Performance estimate : {} pm {}'.format(np.mean(rewards), np.std(rewards)))


    def make_video(self, replay, ext=''):
        n_frames = len(replay)
        b_s, n_channels, n_w, n_h = replay[0].shape
        writer = VideoWriter(self.filename+ext+'.mp4')
        for i in range(n_frames):
            writer.writeFrame(replay[i][0][[1, 2, 0]]*255)
        writer.close()

    def save(self, ext=''):
        torch.save(self.policy_net.state_dict(), self.filename+ext+'.pol.ckpt')
        torch.save(self.target_net.state_dict(), self.filename+ext+'.tgt.ckpt')

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load('./trained_agents/'+filename+'.pol.ckpt'))
        self.target_net.load_state_dict(torch.load('./trained_agents/'+filename+'.tgt.ckpt'))
