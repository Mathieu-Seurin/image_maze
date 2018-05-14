import shutil
import hashlib
import json
import os

import logging
from logging.handlers import RotatingFileHandler

def override_config_recurs(config, config_extension):
    try:
        config_extension['name'] = config['name']+'_'+config_extension['name']
    except KeyError:
        pass

    for key, value in config_extension.items():
        if type(value) is dict:
            config[key] = override_config_recurs(config[key], config_extension[key])
        else:
            config[key] = value

    return config

def load_single_config(config_file):
    with open(config_file, 'rb') as f_config:
        config_str = f_config.read()
        config = json.loads(config_str.decode('utf-8'))
    return config

def load_config_and_logger(env_config_file, model_config_file, exp_dir,
                           args=None,
                           env_ext_file=None,
                           model_ext_file=None):

    # Load env file and model
    env_config = load_single_config(env_config_file)
    model_config = load_single_config(model_config_file)

    # Override env and model files if specified
    if env_ext_file is not None:
        env_ext_config = load_single_config(env_ext_file)
        env_config = override_config_recurs(env_config, env_ext_config)
    if model_ext_file is not None:
        model_ext_config = load_single_config(model_ext_file)
        model_config = override_config_recurs(model_config, model_ext_config)

    # Merge env and model config into one dict
    env_config['env_name'] = env_config['name']
    env_config.update(model_config)
    config = env_config

    # Compute unique identifier based on those configs
    config_byte = json.dumps(config).encode()
    exp_identifier = hashlib.md5(config_byte).hexdigest()

    save_path = '{}/{{}}'.format(os.path.join(exp_dir, config['env_name'], exp_identifier))
    if not os.path.isdir(save_path.format('')):
        os.makedirs(save_path.format(''))

    # Write which config files were used, in case the names in config are not set
    with open(save_path.format("config_files.txt"), "w") as f:
        f.write(env_config_file)
        if env_ext_file:
            f.write(env_config_file)

    # Create logger
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
    shutil.copy(env_config_file, save_path.format('config.json'))

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

def set_seed(config, parsed_args=None):

    import numpy as np
    import torch
    import random
    if parsed_args is None:
        seed = config["seed"]
    else:
        print('Using seed {} from parser argument'.format(parsed_args.seed))
        seed = parsed_args.seed
    if seed > -1:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

def write_seed_extensions(seed_range, out_name='../config/seed_extensions/'):
    for seed in seed_range:
        with open(out_name + str(seed), 'w+', encoding="utf8") as f_extension:
            json.dump({"seed": seed}, f_extension)




if __name__ == "__main__":

    print("TESTING EXTENSION")

    config_location = "config/unittest_config/{}"
    config_file = config_location.format("base_test.json")
    extension_file = config_location.format("extension_test.json")
    ext, hashed_id = load_config_extended(config_file=config_file,
                         extension_file=extension_file)

    print(ext)
