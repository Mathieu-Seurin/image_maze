import subprocess
from subprocess import Popen

import multiprocessing as mp
mp = mp.get_context('spawn')


from itertools import product
import os
import time
from parse_dir import parse_env_subfolder, plot_best, plot_best_per_model

from main import full_train_test

import argparse
import json

def create_launch_proc(command_param, device_to_use):
    env_folder = "config/env_base/"
    env_ext_folder = "config/env_ext/"

    model_folder = "config/model/"
    model_ext_folder = "config/model_ext/"

    exp_dir = 'out/'

    env_file = env_folder+command_param['env']+'.json'
    env_ext_file = env_ext_folder+command_param['env_ext']+'.json'
    model_config = model_folder+command_param['model']+'.json'
    model_ext = model_ext_folder+command_param['model_ext']+'.json'
    seed = command_param['seed']

    # need to be the same order as full_train_test
    # full_train_test(env_config, model_config, env_extension, model_extension, exp_dir, seed=0, device=-1, args=None):
    p = mp.Process(target=full_train_test, args=[env_file, model_config, env_ext_file, model_ext, exp_dir, seed, device_to_use])
    p.start()
    return p

def load_config_file(file):

    run_multiple_config_path = os.path.join('config/run_multiple_config', file)
    config = json.load(open(run_multiple_config_path, 'r'))
    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-config_multiple_file", type=str, default="test.json", help="config file where you put your schedule")
    args = parser.parse_args()

    config = load_config_file(args.config_multiple_file)

    model_to_test = config['model_to_test']
    extension_to_test = config['extension_to_test']
    env_config = config['env_config']
    env_ext = config['env_ext']
    gpu_available = config['gpu_available']
    capacity_per_gpu = config['capacity_per_gpu']
    n_seed = config['n_seed']

    #extension_to_test = ["small_text_part", "bigger_text_part", "bigger_vision", "smaller_everything", "no_head"]

    n_gpu = len(gpu_available)

    n_processes = capacity_per_gpu * n_gpu
    seeds = [i for i in range(n_seed)]

    command_keys = ['env', 'env_ext', 'model', 'model_ext', 'seed']
    all_commands = [dict(zip(command_keys, command_param)) for command_param in product(env_config, env_ext, model_to_test, extension_to_test, seeds)]
    print("{} experiments to run.".format(len(all_commands)))
    processes = []


    command_remains = len(all_commands) > 0
    for expe_num in range(n_gpu * capacity_per_gpu):

        # Launch expe, fill all processes
        try:
            command_param = all_commands.pop()
        except IndexError:
            command_remains = False
            break

        print("Launching new expe, {} remains".format(len(all_commands)))
        device_to_use = gpu_available[expe_num % n_gpu]
        processes.append(create_launch_proc(command_param=command_param,
                                            device_to_use=device_to_use))

    while command_remains:
        for p_num, p in enumerate(processes):

            if not p.is_alive():

                try:
                    command_param = all_commands.pop()
                    print("Launching new exp, {} remaining".format(len(all_commands)))
                except IndexError:
                    command_remains = False
                    break

                device_to_use = gpu_available[p_num % n_gpu]
                new_p = create_launch_proc(command_param=command_param, device_to_use=device_to_use)
                processes[p_num] = new_p

        time.sleep(2)

    for expe in processes:
        expe.join()

    print('Done running experiments')

    for env_str, env_ext_str in product(env_config, env_ext):
        out_dir = "out/" + env_str + '_' + env_ext_str
        parse_env_subfolder(out_dir=out_dir)
        plot_best(env_dir=out_dir)
        plot_best_per_model(env_dir=out_dir)
