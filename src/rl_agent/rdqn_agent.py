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
from .agent_utils import ReplayMemoryRecurrent, Transition, Flatten, check_params_changed, freeze_as_np_dict, compute_slow_params_update
import pickle

from rl_agent.forward_model_recurrent import DRQNText

import logging
import os

from .gpu_utils import use_cuda, FloatTensor, LongTensor, ByteTensor, Tensor
from image_text_utils import TextToIds

import os

class RDQN_Agent(object):
    def __init__(self, config, n_action, state_dim, is_multi_objective, objective_type):

        self.objective_is_text = 'text' in objective_type

        config = config['rdqn_params']

        if self.objective_is_text:
            model = DRQNText(config=config,
                                  n_actions=n_action,
                                  state_dim=state_dim,
                                  is_multi_objective=is_multi_objective)
            self.text_to_vect = TextToIds()

        else:
            raise NotImplementedError("Not sure it will work from scratch")

        self.forward_model = model
        self.ref_model = deepcopy(self.forward_model)
        if use_cuda:
            self.forward_model.cuda()
            self.ref_model.cuda()
        self.n_action = n_action

        self.memory = ReplayMemoryRecurrent(config["memory_size"], max_seq_length=config["max_seq_length"])
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

        # When the episode is over => store sequence
        self.memory.end_of_ep()

    def train(self):
        self.forward_model.train()
        self.ref_model.train()

    def eval(self):
        self.forward_model.eval()
        self.ref_model.eval()


    def forward(self, state, epsilon=0.1):

        plop = np.random.rand()
        if plop < epsilon:
            idx = np.random.randint(self.n_action)
        else:
            # state is {"env_state" : img, "objective": img/text}
            var_state = dict()
            var_state['env_state'] = Variable(FloatTensor(state['env_state']).unsqueeze(0), volatile=True)

            if self.objective_is_text:
                objective = self.text_to_vect.sentence_to_matrix(state['objective'])
                objective = LongTensor(objective) # Long expected for int input
            else:
                objective = FloatTensor(state['objective']) # for image, use Float

            var_state['objective'] = Variable(objective.unsqueeze(0), volatile=True)
            idx = self.forward_model(var_state).data.max(1)[1].cpu().numpy()[0]

        return idx

    def optimize_one_time_step_batch(self, batch, batch_size):
        """
        DRQN needs to be optimized for several timestep, updating at every time step h_t to train the recurrent part of
        the network.
        :return:
        """
        state_batch = Variable(torch.cat(batch.state).type(Tensor))

        if self.objective_is_text:
            # Batchify : all sequence must have the same size, so you pad the dialogue with token
            objective_batch = self.text_to_vect.pad_batch_sentence(batch.objective)
            objective_batch = Variable(torch.cat(objective_batch).type(LongTensor))

        else:
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
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]).type(Tensor),
                                         volatile=True)

        if self.objective_is_text:
            objective_batch = [batch.objective[i] for i, s in enumerate(batch.next_state) if s is not None]
            objective_batch = self.text_to_vect.pad_batch_sentence(objective_batch)
            non_final_state_corresponding_objective = torch.cat(objective_batch).type(LongTensor)
        else:
            non_final_state_corresponding_objective = torch.cat(
                [batch.objective[i] for i, s in enumerate(batch.next_state) if s is not None]).type(Tensor)

        non_final_state_corresponding_objective = Variable(non_final_state_corresponding_objective, volatile=True)

        non_final_next_states_obj = dict()
        non_final_next_states_obj['env_state'] = non_final_next_states
        non_final_next_states_obj['objective'] = non_final_state_corresponding_objective

        if len(non_final_next_states.shape) == 3:
            non_final_next_states = non_final_next_states.unsqueeze(0)

        next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
        next_state_values[non_final_mask] = self.ref_model(non_final_next_states_obj).max(1)[0]

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values)
        return loss

    def optimize(self, state, action, next_state, reward):

        # state is {"env_state" : img, "objective": img}
        state_loc = FloatTensor(state['env_state'])
        next_state_loc = FloatTensor(next_state['env_state'])

        if self.objective_is_text:
            objective = self.text_to_vect.sentence_to_matrix(state['objective'])
        else:
            objective = state['objective']

        objective = FloatTensor(objective).unsqueeze(0)

        # if self.concatenate_objective:
        #     state_loc = torch.cat((state_loc, FloatTensor(state['objective'])))
        #     next_state_loc = torch.cat((next_state_loc, FloatTensor(next_state['objective'])))

        state = state_loc.unsqueeze(0)
        next_state = next_state_loc.unsqueeze(0)
        action = LongTensor([int(action)]).view((1, 1,))
        reward = FloatTensor([reward])

        self.memory.push(state, action, next_state, reward, objective)

        transitions, batch_size = self.memory.sample(self.batch_size)
        if not batch_size:
            # not enough sample at the moment
            return


        loss = 0
        self.forward_model.optimize_mode(optimize=True, batch_size=batch_size)
        self.ref_model.optimize_mode(optimize=True, batch_size=batch_size)

        for timestep_batch in transitions:

            batch = Transition(*zip(*timestep_batch))
            current_loss = self.optimize_one_time_step_batch(batch, batch_size)
            loss += current_loss

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

        new_params = freeze_as_np_dict(self.forward_model.state_dict())
        #check_params_changed(old_params, new_params)

        self.forward_model.optimize_mode(optimize=False)
        self.ref_model.optimize_mode(optimize=False)

        return loss.data[0]


    def save_state(self):
        # Store the whole agent state somewhere
        state_dict = self.forward_model.state_dict()
        memory = deepcopy(self.memory)
        return state_dict, memory

    def load_state(self, state_dict, memory):
        self.forward_model.load_state_dict(state_dict)
        self.memory = deepcopy(memory)
