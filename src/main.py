import argparse
import logging
import time
from itertools import count

from maze import ImageGridWorld
from rl_agent.basic_agent import AbstractAgent
from config import load_config_and_logger

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
env = ImageGridWorld(config=config["env_type"],
                     show=True)
# Agent
# =====
rl_agent = AbstractAgent(config, env.action_space()) # Todo : agent factory that loads the good agent based on config file


n_episode = config["optim"]["n_episode"]

for ep in range(n_episode):
    observation = env.reset()
    done = False
    count = 0

    if args.display:
        env.render()

    logging.info(env.position)
    while not done:
        action = rl_agent.forward(observation)
        count += 1
        logging.info(action)
        observation, reward, done, info = env.step(action)
        if args.display:
            env.render()
            time.sleep(2)
        logging.info(info)
        logging.info(done)


        # Todo : reward logger

    logging.info(f"Temps pour trouver la reward : {count}")


