import argparse
import logging
import time
from itertools import count

from maze import ImageGridWorld
from feature_maze import ImageFmapGridWorld

from rl_agent.basic_agent import AbstractAgent
from rl_agent.dqn_agent import DQNAgent
from rl_agent.reinforce_agent import ReinforceAgent
from config import load_config_and_logger, set_seed
import torch.optim as optim
import torch
import numpy as np
from image_utils import make_video, make_eval_plot

parser = argparse.ArgumentParser('Log Parser arguments!')

parser.add_argument("-exp_dir", type=str, default="out", help="Directory with one expe")
parser.add_argument("-env_config", type=str, help="Which file correspond to the experiment you want to launch ?")
parser.add_argument("-model_config", type=str, help="Which file correspond to the experiment you want to launch ?")
parser.add_argument("-env_extension", type=str, help="Do you want to override parameters in the env file ?")
parser.add_argument("-model_extension", type=str, help="Do you want to override parameters in the model file ?")
parser.add_argument("-display", type=str, help="Display images or not")
parser.add_argument("-seed", type=int, default=0, help="Manually set seed when launching exp")

args = parser.parse_args()
# Load_config also creates logger inside (INFO to stdout, INFO to train.log)
config, exp_identifier, save_path = load_config_and_logger(env_config_file=args.env_config,
                                                           model_config_file=args.model_config,
                                                           env_ext_file=args.env_extension,
                                                           model_ext_file=args.model_extension,
                                                           args=args,
                                                           exp_dir=args.exp_dir
                                                           )

logging = logging.getLogger()
set_seed(config, args)


env = ImageFmapGridWorld(config=config["env_type"])

if config["agent_type"] == 'random':
    rl_agent = AbstractAgent(config, env.action_space())
elif 'dqn' in config["agent_type"]:
    rl_agent = DQNAgent(config, env.action_space(), env.state_objective_dim(), env.is_multi_objective)
elif config["agent_type"] == 'reinforce':
    rl_agent = ReinforceAgent(config['reinforce_params'], env.action_space())
else:
    assert False, "Wrong agent type : {}".format(config["agent_type"])


n_epochs = config["train_params"]["n_epochs"]
epsilon_schedule = config["train_params"]["epsilon_schedule"][0]
epsilon_init = config["train_params"]["epsilon_schedule"][1]
test_every = config["train_params"]["test_every"]
n_epochs_test = config["train_params"]["n_epochs_test"]

verbosity = config["io"]["verbosity"]


def train(agent, env):
    if epsilon_schedule == 'linear':
        eps_range = np.linspace(epsilon_init, 0., n_epochs)
    elif epsilon_schedule=='constant':
        eps_range = [epsilon_init for _ in range(n_epochs)]
    elif epsilon_schedule=='exp':
        eps_decay = n_epochs / 4.
        eps_range = [epsilon_init * np.exp(-1. * i / eps_decay) for i in range(n_epochs)]

    logging.info(" ")
    logging.info("Begin Training")
    logging.info("===============")

    for epoch in range(n_epochs):
        state = env.reset(show=False)
        done = False
        time_out = 20
        num_step = 0

        if epoch % test_every == 0:
            reward, length = test(agent, env, config, epoch)
            logging.info("Epoch {} test : averaged reward {:.2f}, average length {:.2f}".format(epoch, reward, length))

            with open(save_path.format('train_lengths'), 'a+') as f:
                f.write("{} {}\n".format(epoch, length))
            with open(save_path.format('train_rewards'), 'a+') as f:
                    f.write("{} {}\n".format(epoch, reward))
            make_eval_plot(save_path.format('train_lengths'), save_path.format('eval_curve.png'))

        while not done and num_step < time_out:
            num_step += 1
            action = agent.forward(state, eps_range[epoch])
            next_state, reward, done, _ = env.step(action)
            loss = agent.optimize(state, action, next_state, reward)
            state = next_state

        agent.callback(epoch)




def test(agent, env, config, num_test):

    # Setting the model into test mode (for dropout for example)
    agent.eval()

    lengths, rewards = [], []
    obj_type = config['env_type']['objective']['type']
    number_epochs_to_store = config['io']['num_epochs_to_store']

    if obj_type == 'fixed':
        test_objectives = [env.reward_position]
    elif 'image' in obj_type:
        # For now, test only on previously seen examples
        test_objectives = env.objectives
    else:
        assert False, 'Objective {} not supported'.format(obj_type)

    for num_objective, objective in enumerate(test_objectives):
        logging.debug('Switching objective to {}'.format(objective))
        env.reward_position = objective

        for epoch in range(n_epochs_test):

            # WARNING FREEZE COUNT SO THE MAZE DOESN'T CHANGE
            env.count_ep_in_this_maze = 0
            env.count_current_objective = 0

            state = env.reset(show=False)

            done = False
            time_out = 20
            num_step = 0
            epoch_rewards = []
            video = []

            if epoch < number_epochs_to_store:
                video.append(env.render(show=False))

            while not done and num_step < time_out:
                num_step += 1
                action = agent.forward(state, 0.)
                next_state, reward, done, _ = env.step(action)

                if epoch < number_epochs_to_store:
                    video.append(env.render(show=False))

                epoch_rewards += [reward]
                state = next_state

            rewards.append(np.sum(epoch_rewards))
            lengths.append(len(epoch_rewards))

            if epoch < number_epochs_to_store:
                make_video(video, save_path.format('test_{}_{}_{}'.format(num_test, num_objective, epoch)))

    # Setting the model back into train mode (for dropout for example)
    agent.train()

    return np.mean(rewards), np.mean(lengths)

if config['agent_type'] != 'random':
    train(rl_agent, env)
else:
    print(test(rl_agent, env))
