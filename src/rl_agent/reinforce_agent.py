"""
Gym complient implementation of pytorch-based agents
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from .agent_utils import check_params_changed, freeze_as_np_dict
from .dqn_models import SoftmaxDQN
import logging
from copy import deepcopy

# logging.basicConfig(level=logging.DEBUG)

# GPU compatibility setup
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ReinforceAgent(object):
    def __init__(self, config, n_action):
        self.forward_model = SoftmaxDQN(config, n_action).cuda()
        self.n_action = n_action
        self.gamma = 0.99
        self.saved_log_probs_epoch = []
        self.rewards_epoch = []
        self.last_loss = np.nan
        self.policy_loss = []
        self.rewards = []

    def apply_config(self, config):
        pass

    def callback(self, epoch):
        R = 0
        rewards = []

        for r in self.rewards_epoch[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        for log_prob, reward in zip(self.saved_log_probs_epoch, rewards):
            self.policy_loss.append(-log_prob * reward)

        self.saved_log_probs_epoch = []
        self.rewards_epoch = []

        update_every = 32
        if epoch % update_every == 0:
            logging.info('Epoch {} : mean reward {}'.format(epoch, np.sum(self.rewards) / update_every))
            self.rewards = []
            old_params = freeze_as_np_dict(self.forward_model.state_dict())
            self.forward_model.optimizer.zero_grad()
            policy_loss = torch.cat(self.policy_loss).sum()
            policy_loss.backward()
            for param in self.forward_model.parameters():
                logging.debug(param.grad.data.sum())
                param.grad.data.clamp_(-1., 1.)
            self.forward_model.optimizer.step()
            new_params = freeze_as_np_dict(self.forward_model.state_dict())
            check_params_changed(old_params, new_params)

            self.last_loss = deepcopy(policy_loss.data[0])
            self.policy_loss = []


    def forward(self, state, epsilon=0.1):
        # Epsilon has no influence, keep it for compatibility
        state = state['env_state']
        state = FloatTensor(state)
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        probs = self.forward_model(Variable(state).type(FloatTensor))
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs_epoch.append(m.log_prob(action))
        return action.data[0]

    def optimize(self, state, action, next_state, reward, batch_size=16):
        self.rewards_epoch.append(reward)
        self.rewards.append(reward)
        return self.last_loss
