from subprocess import Popen
from itertools import product
import os

from parse_dir import parse_env_subfolder

env_config = "multi_obj"
env_ext = "10obj_every2"

model_to_test = ['resnet_dqn', 'dqn_filmed']
#extension_to_test = ['soft_update0_1', 'soft_update0_01']
extension_to_test = ['soft_update0_1', 'soft_update0_01', 'soft_update0_001', 'soft_update0_0001',
                      'hard_update0_1', 'hard_update0_01', 'hard_update0_001']

n_seed = 5
n_gpu = 4

seeds = [i for i in range(n_seed)]

for seed in seeds:
    processes = []

    for test_num, (model, model_ext) in enumerate(product(model_to_test,extension_to_test)):

        if test_num > 4*n_gpu - 1:
            "Can't run more than {} expes in parallel".format(n_gpu*4 -1)
            break

        device_to_use = test_num%n_gpu

        config_location = 'config/{}/{}'
        env_config_path = config_location.format("env_base", env_config)
        env_ext_path = config_location.format("env_ext", env_ext)

        model_config_path = config_location.format("model", model)
        model_ext_path = config_location.format("model_ext", model_ext)

        command = "python3 src/main.py -env_config {}.json -env_extension {}.json -model_config {}.json -model_extension {}.json -device {} -seed {}".format(
                    env_config_path, env_ext_path, model_config_path, model_ext_path, device_to_use, seed)
        print(command)
        processes.append(Popen(command, shell=True, env=os.environ.copy()))

    for p in processes:
        p.wait()

print('Done running experiments')

out_dir = "out/"+env_config+'_'+env_ext

parse_env_subfolder(out_dir=out_dir)