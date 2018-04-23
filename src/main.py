import argparse
import logging
import time
from itertools import count

from maze import ImageGridWorld
from rl_agent.basic_agent import AbstractAgent
from rl_agent.dqn_agent import DQNAgent
from rl_agent.reinforce_agent import ReinforceAgent
from config import load_config_and_logger
import torch.optim as optim
import torch
import numpy as np
from image_utils import make_video, make_eval_plot

parser = argparse.ArgumentParser('Log Parser arguments!')

parser.add_argument("-exp_dir", type=str, default="out", help="Directory with one expe")
parser.add_argument("-config", type=str, help="Which file correspond to the experiment you want to launch ?")
parser.add_argument("-extension", type=str, help="Do you want to override parameters in the config file ?")
parser.add_argument("-display", type=str, help="Display images or not")

args = parser.parse_args()
# Load_config also creates logger inside (INFO to stdout, INFO to train.log)
config, exp_identifier, save_path = load_config_and_logger(config_file=args.config,
                    exp_dir=args.exp_dir, args=args, extension_file=args.extension)
logging = logging.getLogger()

env = ImageGridWorld(config=config["env_type"], show=False)

if config["agent_type"] == 'random':
    rl_agent = AbstractAgent(config, env.action_space())
elif config["agent_type"] == 'dqn':
    rl_agent = DQNAgent(config['dqn_params'], env.action_space())
elif config["agent_type"] == 'reinforce':
    rl_agent = ReinforceAgent(config['dqn_params'], env.action_space())
else:
    assert False, "Wrong agent type : {}".format(config["agent_type"])

n_epochs = config["train_params"]["n_epochs"]
batch_size = config["train_params"]["batch_size"]
epsilon_schedule = config["train_params"]["epsilon_schedule"][0]
epsilon_init = config["train_params"]["epsilon_schedule"][1]
test_every = config["train_params"]["test_every"]
n_epochs_test = config["train_params"]["n_epochs_test"]

verbosity = config["io"]["verbosity"]
gif_verbosity = config["io"]["gif_verbosity"]


def train(agent, env):
    if epsilon_schedule == 'linear':
        eps_range = np.linspace(epsilon_init, 0., n_epochs)
    elif epsilon_schedule=='constant':
        eps_range = [epsilon_init for _ in range(n_epochs)]
    elif epsilon_schedule=='exp':
        eps_decay = n_epochs / 4.
        eps_range = [epsilon_init * np.exp(-1. * i / eps_decay) for i in range(n_epochs)]

    for epoch in range(n_epochs):
        state = env.reset(show=False)
        done = False
        video = []
        time_out = 20
        time = 0

        while not done and time < time_out:
            time += 1

            if gif_verbosity != 0:
                if epoch % gif_verbosity == 0 and epoch != 0:
                    video.append(env.render(display=False))

            action = agent.forward(state, eps_range[epoch])
            next_state, reward, done, _ = env.step(action)
            loss = agent.optimize(state, action, next_state, reward, batch_size=batch_size)
            state = next_state

        agent.callback(epoch)

        if epoch % test_every == 0:
            reward, length = test(agent, env)
            logging.info("Epoch {} test : averaged reward {}, average length {}".format(epoch, reward, length))
            with open(save_path.format('train_lengths'), 'a+') as f:
                f.write("{} {}\n".format(epoch, length))
            with open(save_path.format('train_rewards'), 'a+') as f:
                    f.write("{} {}\n".format(epoch, reward))
            make_eval_plot(save_path.format('train_lengths'), save_path.format('eval_curve.png'))


        if gif_verbosity != 0:
            if epoch % gif_verbosity == 0 and epoch != 0:
                make_video(video, save_path.format('train_' + str(epoch)))

def test(agent, env):
    lengths, rewards = [], []
    obj_type = config['env_type']['objective']['type']

    if obj_type in ['image', 'image_no_bkg', 'random_image']:
        # For now, test only on previously seen examples
        test_objectives = env.objectives
    elif obj_type == 'fixed':
        test_objectives = [env.reward_position]
    else:
        assert False, 'Objective {} not supported'.format(obj_type)

    for objective in test_objectives:
        logging.debug('Switching objective to {}'.format(objective))
        env.reward_position = objective

        for epoch in range(n_epochs_test):
            state = env.reset(show=False)
            done = False
            time_out = 20
            time = 0
            epoch_rewards = []

            while not done and time < time_out:
                time += 1
                action = agent.forward(state, 0.)
                next_state, reward, done, _ = env.step(action)
                epoch_rewards += [reward]
                state = next_state

            rewards.append(np.sum(epoch_rewards))
            lengths.append(len(epoch_rewards))

    return np.mean(rewards), np.mean(lengths)

if config['agent_type'] != 'random':
    train(rl_agent, env)
else:
    print(test(rl_agent, env))
