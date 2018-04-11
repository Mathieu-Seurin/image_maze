import shutil
import hashlib
import json
import os

import logging
from logging.handlers import RotatingFileHandler

def load_config(config_file, exp_dir, args=None):
    with open(config_file, 'rb') as f_config:
        config_str = f_config.read()
        exp_identifier = hashlib.md5(config_str).hexdigest()
        config = json.loads(config_str.decode('utf-8'))

    save_path = '{}/{{}}'.format(os.path.join(exp_dir, exp_identifier))
    if not os.path.isdir(save_path.format('')):
        os.makedirs(save_path.format(''))

    # create logger
    logger = create_logger(save_path.format('train.log'))
    logger.info("Config Hash {}".format(exp_identifier))
    logger.info("Config name : {}".format(config["name"]))
    logger.info(config)

    if args is not None:
        for key, val in vars(args).items():
            logger.info("{} : {}".format(key, val))

    # set seed
    set_seed(config)

    # copy config file
    shutil.copy(config_file, save_path.format('config.json'))

    return config, exp_identifier, save_path

def create_logger(save_path):
    logger = logging.getLogger()
    # Debug = write everything
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    file_handler = RotatingFileHandler(save_path, 'a', 1000000, 1)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger

def set_seed(config):
    import numpy as np
    import torch
    seed = config["seed"]
    if seed > -1:
        np.random.seed(seed)
        torch.manual_seed(seed)