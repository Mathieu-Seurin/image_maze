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
from .agent_utils import ReplayMemory, Transition, Flatten, check_params_changed, freeze_as_np_dict, compute_slow_params_update
from .dqn_models import DQN
import pickle
from rl_agent.FiLM_agent import FilmedNet

import logging
import os

from .gpu_utils import use_cuda, FloatTensor, LongTensor, ByteTensor, Tensor

import os

class DQNAgent(object):
    def __init__(self, config, n_action, state_dim, is_multi_objective):

        if config['agent_type'] == "resnet_dqn":
            config = config['resnet_dqn_params']
            model = FilmedNet(config=config,
                              n_actions=n_action,
                              state_dim=state_dim,
                              is_multi_objective=is_multi_objective)

        elif config["agent_type"] == "dqn":
            config = config['dqn_params']
            model = DQN(config=config,
                        n_action=n_action,
                        state_dim=state_dim,
                        is_multi_objective=is_multi_objective)
        else:
            raise NotImplementedError("model_type not recognized")

        self.forward_model = model
        self.ref_model = deepcopy(self.forward_model)
        if use_cuda:
            self.forward_model.cuda()
            self.ref_model.cuda()
        self.n_action = n_action

        self.memory = ReplayMemory(4096)
        self.discount_factor = self.forward_model.discount_factor

        self.tau = config['tau']
        self.batch_size = config["batch_size"]
        self.soft_update = config["soft_update"]

        # logging.info('Model summary :')
        # logging.info(self.forward_model.forward)

    def apply_config(self, config):
        pass

    def callback(self, epoch):
        if not self.soft_update and epoch % int(1/self.tau) == 0:
            self.ref_model.load_state_dict(self.forward_model.state_dict())

    def train(self):
        self.forward_model.train()
        self.ref_model.train()

    def eval(self):
        self.forward_model.eval()
        self.ref_model.eval()

    def forward(self, state, epsilon=0.1):

        # if self.concatenate_objective:
        #     state_loc = torch.cat((state_loc, FloatTensor(state['objective'])))
        plop = np.random.rand()
        if plop < epsilon:
            idx = np.random.randint(self.n_action)
        else:
            # state is {"env_state" : img, "objective": img}
            var_state = dict()
            var_state['env_state'] = Variable(FloatTensor(state['env_state']).unsqueeze(0), volatile=True)
            var_state['objective'] = Variable(FloatTensor(state['objective']).unsqueeze(0), volatile=True)
            idx = self.forward_model(var_state).data.max(1)[1].cpu().numpy()[0]

        return idx

    def optimize(self, state, action, next_state, reward):

        # state is {"env_state" : img, "objective": img}
        state_loc = FloatTensor(state['env_state'])
        next_state_loc = FloatTensor(next_state['env_state'])
        objective = FloatTensor(state['objective'])

        # if self.concatenate_objective:
        #     state_loc = torch.cat((state_loc, FloatTensor(state['objective'])))
        #     next_state_loc = torch.cat((next_state_loc, FloatTensor(next_state['objective'])))

        objective = objective.unsqueeze(0)
        state = state_loc.unsqueeze(0)
        next_state = next_state_loc.unsqueeze(0)
        action = LongTensor([int(action)]).view((1, 1,))
        reward = FloatTensor([reward])

        self.memory.push(state, action, next_state, reward, objective)

        if len(self.memory.memory) < self.batch_size:
            batch_size = len(self.memory.memory)
        else:
            batch_size = self.batch_size

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(batch.state).type(Tensor))
        objective_batch = Variable(torch.cat(batch.objective).type(Tensor))

        action_batch = Variable(torch.cat(batch.action).type(LongTensor))
        reward_batch = Variable(torch.cat(batch.reward).type(Tensor))

        if len(state_batch.shape) == 3:
            state_batch = state_batch.unsqueeze(0)
            objective_batch.unsqueeze(0)

        state_obj = dict()
        state_obj['env_state'] = state_batch
        state_obj['objective'] = objective_batch

        state_action_values = self.forward_model(state_obj).gather(1, action_batch)

        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]).type(Tensor), volatile=True)
        non_final_state_corresponding_objective = Variable(torch.cat([batch.objective[i] for i, s in enumerate(batch.next_state) if s is not None]).type(Tensor), volatile=True)

        non_final_next_states_obj = dict()
        non_final_next_states_obj['env_state'] = non_final_next_states
        non_final_next_states_obj['objective'] = non_final_state_corresponding_objective


        if len(non_final_next_states.shape) == 3:
            non_final_next_states = non_final_next_states.unsqueeze(0)

        next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
        next_state_values[non_final_mask] = self.ref_model(non_final_next_states_obj).max(1)[0]

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values)

        #old_params = freeze_as_np_dict(self.forward_model.state_dict())
        self.forward_model.optimizer.zero_grad()
        loss.backward()
        for param in self.forward_model.parameters():
            logging.debug(param.grad.data.sum())
            param.grad.data.clamp_(-1., 1.)
        self.forward_model.optimizer.step()

        # Update slowly ref model towards fast model, to stabilize training.
        if self.soft_update:
            self.ref_model.load_state_dict(compute_slow_params_update(self.ref_model, self.forward_model, self.tau))

        #new_params = freeze_as_np_dict(self.forward_model.state_dict())
        #check_params_changed(old_params, new_params)
        return loss.data[0]


    def save_state(self):
        # Store the whole agent state somewhere
        state_dict = self.forward_model.state_dict()
        memory = deepcopy(self.memory)
        return state_dict, memory

    def load_state(self, state_dict, memory):
        self.forward_model.load_state_dict(state_dict)
        self.memory = deepcopy(memory)
