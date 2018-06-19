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
from .dqn_models import DQN, SoftmaxDQN
import logging
from copy import deepcopy
from rl_agent.FiLM_agent import FilmedNet, FilmedNetText
import os
from image_text_utils import TextToIds


# GPU compatibility setup
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# logging.basicConfig(level=logging.DEBUG)

# Suggestions for parameter tuning (based on exps for fixed maze, 5 obj every 2:

# - large "update_every" parameter (200 for ex) seems to be better : give
# the agent more trajectories to learn from, improves stability
# - few epochs with each obj (2) seems good to avoid local minima
# - consequently, changing the maze images often should also be good
# - lr of 10e-3 seems to be the sweet spot
# - increase this as necessary for more


class ReinforceAgent(object):
    def __init__(self, config, n_action, state_dim, is_multi_objective, objective_type):
        self.objective_is_text = 'text' in objective_type
        if config['agent_type'] == "resnet_reinforce":
            config = config['resnet_reinforce_params']
            if self.objective_is_text:
                model = FilmedNetText(config=config,
                                      n_actions=n_action,
                                      state_dim=state_dim,
                                      is_multi_objective=is_multi_objective)
                self.text_to_vect = TextToIds()

            else:
                model = FilmedNet(config=config,
                                  n_actions=n_action,
                                  state_dim=state_dim,
                                  is_multi_objective=is_multi_objective)

        elif config["agent_type"] == "reinforce":
            config = config['resnet_reinforce_params']
            model = DQN(config=config,
                        n_action=n_action,
                        state_dim=state_dim,
                        is_multi_objective=is_multi_objective)
        else:
            raise NotImplementedError("agent_type not recognized")


        self.forward_model = model
        if use_cuda:
            self.forward_model.cuda()
        self.n_action = n_action
        self.gamma = config['discount_factor']
        self.update_every = config['update_every']
        self.entropy_penalty = config['entropy_penalty']
        self.concatenate_objective = config['concatenate_objective']
        self.last_loss = np.nan
        self.rewards_epoch = []
        self.rewards_replay = []
        self.states_epoch = []
        self.actions_epoch = []
        self.states_replay = []
        self.actions_replay = []

        logging.info('Model summary :')
        logging.info(self.forward_model.forward)

    def apply_config(self, config):
        pass

    def callback(self, epoch):
        R = 0
        rewards = []

        for r in self.rewards_epoch[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        self.rewards_replay.extend(rewards)
        self.states_replay.extend(self.states_epoch)
        self.actions_replay.extend(self.actions_epoch)
        self.rewards_epoch = []
        self.states_epoch = []
        self.actions_epoch = []


        if epoch % self.update_every == 0 and epoch != 0:
            self.win_history = []

            rewards = Tensor(self.rewards_replay)
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
            actions = LongTensor(self.actions_replay)

            env_states = Variable(torch.cat([s['env_state'] for s in self.states_replay]))

            if self.objective_is_text:
                # Batchify : all sequence must have the same size, so you pad the dialogue with token
                objective_batch = [s['objective'] for s in self.states_replay]
                objective_batch = self.text_to_vect.pad_batch_sentence(objective_batch)
                objectives = Variable(torch.cat(objective_batch).type(LongTensor))
            else:
                objectives = Variable(torch.cat([s['objective'] for s in self.states_replay]))

            # print(env_states.shape, objectives.shape)
            # print(env_states.shape, objectives.shape, actions.shape)
            states = {'env_state': env_states, 'objective': objectives}


            full_log_probs = torch.log(F.softmax(self.forward_model(states), dim=1))
            entropy_loss = (torch.exp(full_log_probs) * full_log_probs).sum(dim=1).sum()


            log_probs = full_log_probs.gather(1, Variable(actions.view(-1, 1))).squeeze(1)
            policy_loss = []

            for log_prob, reward in zip(log_probs, rewards):
                policy_loss.append(-log_prob * reward)

            # print(policy_loss)
            self.rewards_replay = []
            self.actions_replay = []
            self.states_replay = []

            old_params = freeze_as_np_dict(self.forward_model.state_dict())

            try:
                policy_loss = torch.cat(policy_loss).sum()
            except RuntimeError:
                logging.warning('Invalid loss encountered')
                return

            # print(policy_loss, entropy_loss)
            loss = policy_loss + self.entropy_penalty * entropy_loss

            self.forward_model.optimizer.zero_grad()
            loss.backward()
            for param in self.forward_model.parameters():
                logging.debug(param.grad.data.sum())
                param.grad.data.clamp_(-1., 1.)
            self.forward_model.optimizer.step()
            new_params = freeze_as_np_dict(self.forward_model.state_dict())
            check_params_changed(old_params, new_params)

            self.last_loss = deepcopy(policy_loss.data[0])
            self.policy_loss = []

    def eval(self):
        self.forward_model.eval()

    def train(self):
        self.forward_model.eval()

    def forward(self, state, epsilon=0.1, already_embed=False):
        # Epsilon has no influence, keep it for compatibility
        state_loc = dict()
        state_loc['env_state'] = Variable(FloatTensor(state['env_state']).unsqueeze(0), volatile=True)

        if self.objective_is_text:
            if not already_embed:
                objective = self.text_to_vect.sentence_to_matrix(state['objective'])
                objective = LongTensor(objective) # Long expected for int input
            else:
                objective = state['objective']
        else:
            objective = FloatTensor(state['objective']) # for image, use Float
        state_loc['objective'] = Variable(objective.unsqueeze(0), volatile=True)

        probs = F.softmax(self.forward_model(state_loc), dim=1)
        m = Categorical(probs)
        action = m.sample()
        return action.data[0]

    def optimize(self, state, action, next_state, reward, batch_size=16):
        # Just store all relevant info to do batch learning at end of epoch
        self.rewards_epoch.append(reward)
        self.actions_epoch.append(action)
        state_loc = dict()
        state_loc['env_state'] = FloatTensor(state['env_state']).unsqueeze(0)
        if self.objective_is_text:
            objective = self.text_to_vect.sentence_to_matrix(state['objective'])
            objective = LongTensor(objective) # Long expected for int input
        else:
            objective = FloatTensor(state['objective']) # for image, use Float
        state_loc['objective'] = objective.unsqueeze(0)
        self.states_epoch.append(state_loc)
        return self.last_loss

    def save_state(self):
        return self.forward_model.state_dict(), None

    def load_state(self, state_dict, memory):
        # Don't care about memory here
        self.forward_model.load_state_dict(state_dict)
