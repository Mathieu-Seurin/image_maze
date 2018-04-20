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
from image_utils import make_video

parser = argparse.ArgumentParser('Log Parser arguments!')

parser.add_argument("-exp_dir", type=str, default="out", help="Directory with one expe")
parser.add_argument("-config", type=str, help="Which file correspond to the experiment you want to launch ?")
parser.add_argument("-extension", type=str, help="Do you want to override parameters in the config file ?")
parser.add_argument("-display", type=str, help="Display images or not")


# Args parser, logging, config
# ===========================
args = parser.parse_args()
# Load_config also creates logger inside (INFO to stdout, INFO to train.log)
config, exp_identifier, save_path = load_config_and_logger(config_file=args.config,
                                                exp_dir=args.exp_dir,
                                                args=args,
                                                extension_file=args.extension)
logging = logging.getLogger()


# Environment
# ===========
env = ImageGridWorld(config=config["env_type"], show=False)
# Agent
# =====

# Todo : agent factory that loads the good agent based on config file

# rl_agent = AbstractAgent(config, env.action_space())
rl_agent = DQNAgent(config['dqn_params'], env.action_space())
# rl_agent = ReinforceAgent(config['dqn_params'], env.action_space())


n_episode = config["optim"]["n_episode"]
verbosity = config["verbosity"]
gif_verbosity = config["gif_verbosity"]
# To be modified for improved config
def train(agent, env, save_path, n_epochs, epsilon_init=1., epsilon_schedule='exp', eps_decay=None, lr=0.001, batch_size=32):
    if epsilon_schedule == 'linear':
        eps_range = np.linspace(epsilon_init, 0., n_epochs)
    elif epsilon_schedule=='constant':
        eps_range = [epsilon_init for _ in range(n_epochs)]
    elif epsilon_schedule=='exp':
        if not eps_decay:
            eps_decay = n_epochs / 2.
        eps_range = [epsilon_init * np.exp(-1. * i / eps_decay) for i in range(n_epochs)]

    losses, rewards = [], []

    for epoch in range(n_epochs):
        state = env.reset(show=False)
        done = False
        epoch_losses = []
        epoch_rewards = []
        video = []
        time_out = 20
        time = 0

        while not done and time < time_out:
            time += 1
            if epoch % gif_verbosity == 0 and epoch != 0:
                video.append(env.render(display=False))
            action = agent.forward(state, eps_range[epoch])
            next_state, reward, done, _ = env.step(action)
            loss = agent.optimize(state, action, next_state, reward, batch_size=batch_size)
            state = next_state


            epoch_losses.append(loss)
            epoch_rewards.append(reward)

        agent.callback(epoch)

        if epoch % verbosity == 0 and epoch != 0:
            logging.info('Epoch {}: loss= {}, reward= {}, duration= {}'.format(
                epoch, np.mean(epoch_losses), np.sum(epoch_rewards), len(epoch_rewards)))
        losses.append(np.mean(epoch_losses))
        rewards.append(np.sum(epoch_rewards))


        if epoch % gif_verbosity == 0 and epoch != 0:
            make_video(video, save_path.format('train_' + str(epoch)))

            with open(save_path.format('train_losses'), 'a+') as f:
                for l in losses:
                    f.write(str(l)+'\n')
            losses = []
            with open(save_path.format('train_rewards'), 'a+') as f:
                for r in rewards:
                    f.write(str(r)+'\n')
            rewards = []

def test(agent, env, n_epochs, display=False):
    lengths, rewards = [], []

    for epoch in range(n_epochs):
        state = env.reset(show=display)
        done = False
        video = []
        time_out = 20
        time = 0
        epoch_rewards = []

        while not done and time < time_out:
            time += 1
            if epoch % 10 == 1:
                video.append(env.render(display=False))

            action = agent.forward(state, 0.)
            next_state, reward, done, _ = env.step(action)
            epoch_rewards += [reward]
            state = next_state

        logging.info('Epoch {}: reward= {}, duration= {}'.format(
            epoch, np.sum(epoch_rewards), len(epoch_rewards)))

        rewards.append(np.sum(epoch_rewards))
        lengths.append(len(epoch_rewards))

        if epoch % 10 == 1:
            make_video(video, save_path.format('test_' + str(epoch)))

        logging.info('Mean reward :{}, mean duration :{}'.format(np.mean(rewards), np.mean(lengths)))

train(rl_agent, env, save_path, n_episode, batch_size=50)
test(rl_agent, env, 256, display=False)

# For reference, the random agent does 60% exits with mean duration 12
# The optimal is 100% exits and duration 2.31
