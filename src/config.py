import shutil
import hashlib
import json
import os

import logging
from logging.handlers import RotatingFileHandler

def override_config_recurs(config, config_extension):
    for key, value in config_extension.items():
        if type(value) is dict:
            config[key] = override_config_recurs(config[key], config_extension[key])
        else:
            config[key] = value

    return config

def load_config_and_logger(config_file, exp_dir, args=None, extension_file=None):

    if extension_file is None:
        config, exp_identifier = load_config(config_file=config_file)
    else:
        config, exp_identifier = load_config_extended(config_file=config_file,
                                                      extension_file=extension_file)

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


def load_config(config_file):
    with open(config_file, 'rb') as f_config:
        config_str = f_config.read()
        exp_identifier = hashlib.md5(config_str).hexdigest()
        config = json.loads(config_str.decode('utf-8'))

    return config, exp_identifier


def load_config_extended(config_file, extension_file):

    # Load config file
    with open(config_file, 'rb') as f_config:
        config_str = f_config.read()
        config = json.loads(config_str.decode('utf-8'))

    # And its extension to override parameters
    with open(extension_file, 'rb') as f_extension:
        config_str_extension = f_extension.read()
        config_extension = json.loads(config_str_extension.decode('utf-8'))

    #override Base Config with extension parameters
    config = override_config_recurs(config=config,
                                    config_extension=config_extension)

    config_byte = json.dumps(config).encode()
    exp_identifier = hashlib.md5(config_byte).hexdigest()

    return config, exp_identifier


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

if __name__ == "__main__":

    print("TESTING EXTENSION")

    config_location = "config/{}"
    config_file = config_location.format("base_test.json")
    extension_file = config_location.format("extension_test.json")
    load_config_extended(config_file=config_file,
                         extension_file=extension_file,
                         exp_dir="out/",
                         args=None)