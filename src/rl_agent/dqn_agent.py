"""
Gym complient implementation of pytorch-based agents
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
import random
from .agent_utils import ReplayMemory, Transition, Flatten, check_params_changed, freeze_as_np_dict
from .dqn_models import DQN

import logging

# GPU compatibility setup
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class DQNAgent(object):
    def __init__(self, config, n_action):
        self.forward_model = DQN(config, n_action).cuda()
        self.ref_model = deepcopy(self.forward_model).cuda()
        self.n_action = n_action
        self.memory = ReplayMemory(2048)
        self.gamma = 0.99

    def apply_config(self, config):
        pass

    def callback(self, epoch):
        if epoch % 10 == 1:
            self.ref_model.load_state_dict(self.forward_model.state_dict())

    def forward(self, state, epsilon=0.1):
        plop = np.random.rand()
        if plop < epsilon:
            idx = np.random.randint(self.n_action)
        else:
            state = FloatTensor(state)
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            idx = self.forward_model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].cpu().numpy()[0]
        return idx

    def optimize(self, state, action, next_state, reward, batch_size=16):
        state = FloatTensor([state])
        next_state = FloatTensor([next_state])
        action = LongTensor([int(action)]).view((1, 1,))
        reward = FloatTensor([reward])

        self.memory.push(state, action, next_state, reward)
        if len(self.memory.memory) < batch_size:
            batch_size = len(self.memory.memory)

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        if len(state_batch.shape) == 3:
            state_batch = state_batch.unsqueeze(0)

        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)

        state_action_values = self.forward_model(state_batch).gather(1, action_batch)

        if len(non_final_next_states.shape) == 3:
            non_final_next_states = non_final_next_states.unsqueeze(0)

        next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
        next_state_values[non_final_mask] = self.forward_model(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        expected_state_action_values = Variable(expected_state_action_values.data)

        loss = F.mse_loss(state_action_values, expected_state_action_values)


        old_params = freeze_as_np_dict(self.forward_model.state_dict())
        self.forward_model.optimizer.zero_grad()
        loss.backward()
        for param in self.forward_model.parameters():
            logging.debug(param.grad.data.sum())
            param.grad.data.clamp_(-1., 1.)
        self.forward_model.optimizer.step()

        new_params = freeze_as_np_dict(self.forward_model.state_dict())
        check_params_changed(old_params, new_params)
        return loss.data[0]
