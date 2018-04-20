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
        self.gamma = config['gamma']
        self.update_every = config['reinforce_update_every']
        self.concatenate_objective = config['concatenate_objective']
        self.last_loss = np.nan
        self.win_history = []
        self.rewards_epoch = []
        self.rewards_replay = []
        self.states_epoch = []
        self.actions_epoch = []
        self.states_replay = []
        self.actions_replay = []

    def apply_config(self, config):
        pass

    def callback(self, epoch):
        R = 0
        rewards = []

        for r in self.rewards_epoch[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        if np.any(np.array(self.rewards_epoch) > 0):
            self.win_history.append(1)
        else:
            self.win_history.append(0)

        self.rewards_replay.extend(rewards)
        self.states_replay.extend(self.states_epoch)
        self.actions_replay.extend(self.actions_epoch)
        self.rewards_epoch = []
        self.states_epoch = []
        self.actions_epoch = []


        if epoch % self.update_every == 0 and epoch != 0:
            logging.info('Epoch {} : fraction of exits since last update {}'.format(epoch, np.sum(self.win_history) / self.update_every))
            self.win_history = []

            rewards = Tensor(self.rewards_replay)
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

            actions = LongTensor(self.actions_replay)
            states = torch.cat(self.states_replay)
            log_probs = torch.log(self.forward_model(Variable(states))).gather(1, Variable(actions.view(-1, 1))).squeeze(1)

            policy_loss = []
            for log_prob, reward in zip(log_probs, rewards):
                policy_loss.append(-log_prob * reward)

            self.rewards_replay = []
            self.actions_replay = []
            self.states_replay = []

            old_params = freeze_as_np_dict(self.forward_model.state_dict())

            policy_loss = torch.cat(policy_loss).sum()
            self.forward_model.optimizer.zero_grad()
            policy_loss.backward(retain_graph=False)
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
        state_loc = FloatTensor(state['env_state'])
        if self.concatenate_objective:
            state_loc = torch.cat((state_loc, FloatTensor(state['objective'])))
        state_loc = state_loc.unsqueeze(0)
        probs = self.forward_model(Variable(state_loc, volatile=True))
        m = Categorical(probs)
        action = m.sample()
        return action.data[0]

    def optimize(self, state, action, next_state, reward, batch_size=16):
        # Just store all relevant info to do batch learning at end of epoch
        self.rewards_epoch.append(reward)
        self.actions_epoch.append(action)
        state_loc = Tensor(state['env_state'])
        if self.concatenate_objective:
            state_loc = torch.cat((state_loc, FloatTensor(state['objective'])))
        self.states_epoch.append(state_loc.unsqueeze(0))
        return self.last_loss
