import argparse
import logging
import time

from maze import ImageGridWorld
from rl_agent.basic_agent import AbstractAgent
from config import load_config

parser = argparse.ArgumentParser('Log Parser arguments!')

parser.add_argument("-exp_dir", type=str, default="out", help="Directory with one expe")
parser.add_argument("-config", type=str, help="Which file correspond to the experiment you want to launch ?")
# TODO default config and additionnal config.

# Args parser, logging, config
# ===========================
args = parser.parse_args()
# Load_config also creates logger inside (INFO to stdout, INFO to train.log)
config, exp_identifier, save_path = load_config(config_file=args.config, exp_dir=args.exp_dir, args=args)
logging = logging.getLogger()


# Environment
# ===========
env = ImageGridWorld(config=config["env_type"],
                     show=True)
# Agent
# =====
rl_agent = AbstractAgent(config, env.action_space()) # Todo : agent factory that loads the good agent based on config file



n_episode = config["optim"]["n_episode"]

for ep in range(n_episode):
    observation = env.reset()
    done = False
    env.render()
    logging.info(env.position)
    while not done:
        action = rl_agent.forward(observation)
        action=0
        logging.info(action)
        observation, reward, done, info = env.step(action)
        env.render()
        logging.info(info)
        logging.info(done)
        time.sleep(1)
        # Todo : reward logger


