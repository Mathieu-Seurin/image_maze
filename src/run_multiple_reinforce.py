import subprocess
from subprocess import Popen
from itertools import product
import os
import time
from parse_dir import parse_env_subfolder
import gc

test=True

verbose = True
if verbose:
    STDOUT = subprocess.STDOUT
else:
    STDOUT = subprocess.DEVNULL

env_folder = "config/env_base/"
env_ext_folder = "config/env_ext/"

model_folder = "config/model/"
model_ext_folder = "config/model_ext/"


model_to_test = ['reinforce_filmed_pretrain']
extension_to_test = ['reinforce_update_every_1', 'reinforce_update_every_25',
'reinforce_update_every_50', 'reinforce_update_every_100',
'reinforce_update_every_400']

env_config = ["change_maze_10_random_image_no_bkg"]
env_ext = ["5_every_1_reinforce", "10_every_1_reinforce", "15_every_1_reinforce",
            "20_every_1_reinforce"]

n_gpu = 1
capacity_per_gpu = 3
n_seed = 5

if test:
    # I get error when trying to refill the pool of active processes,
    # it seems to try to launch all commands instead of only filling
    # finished ones -> out of memory...
    # Try with only 6 exps total each time

    env_config = ["text_large"]
    model_to_test = ['reinforce_filmed_pretrain']#,
    # model_to_test = ["reinforce_pretrain"]

    # _update_every_25 seems good
    # extension_to_test = ['reinforce_update_every_25']

    # extension_to_test = ['reinforce_entropy_penalty_0', 'reinforce_entropy_penalty_001']
    # extension_to_test = ['reinforce_entropy_penalty_01', 'reinforce_entropy_penalty_02']
    # extension_to_test = ['reinforce_entropy_penalty_005', 'reinforce_entropy_penalty_05']
    # extension_to_test = ['reinforce_entropy_penalty_045', 'reinforce_entropy_penalty_055']
    # extension_to_test = ['reinforce_entropy_penalty_06']#, 'reinforce_entropy_penalty_04']

    extension_to_test = ['reinforce_bigger_text_part']
    # env_config = ["change_maze_10_random_image_no_bkg"]
    env_ext = ["10_every_1_reinforce"]

    n_gpu = 1
    n_seed = 3


seeds = [i for i in range(n_seed)]
general_command = "python3 src/main.py -env_config {}.json -env_extension {}.json -model_config {}.json -model_extension {}.json -device {} -seed {}"

#Fill queue with all your expe
command_keys = ['env', 'env_ext', 'model', 'model_ext', 'seed']
all_commands = [dict(zip(command_keys, command_param)) for command_param in product(env_config, env_ext, model_to_test, extension_to_test, seeds)]
print("{} experiments to run.".format(len(all_commands)))
processes = []

# Launch expe, fill all processes
for expe_num in range(n_gpu*capacity_per_gpu):
    try:
        command_param = all_commands.pop()
    except IndexError:
        break

    env_file = env_folder+command_param['env']
    env_ext_file = env_ext_folder+command_param['env_ext']
    model_config = model_folder+command_param['model']
    model_ext = model_ext_folder+command_param['model_ext']
    seed = command_param['seed']
    device_to_use = int(expe_num%n_gpu)

    command = general_command.format(env_file, env_ext_file, model_config, model_ext, device_to_use, seed)
    processes.append(Popen(command, shell=True, env=os.environ.copy(), stderr=STDOUT))

print("{} expes remains at the moment".format(len(all_commands)))
remains_command = len(all_commands) > 0
print(all_commands)
while remains_command:
    gc.collect()
    time.sleep(1)
    for expe_num, expe in enumerate(processes):
        if expe.poll() is not None:
            try:
                command_param = all_commands.pop()
                print("Launching new expe, {} expe remains".format(len(all_commands)))

            except IndexError:
                remains_command = False
                break

            env_file = env_folder + command_param['env']
            env_ext_file = env_ext_folder + command_param['env_ext']
            model_config = model_folder + command_param['model']
            model_ext = model_ext_folder + command_param['model_ext']
            seed = command_param['seed']
            device_to_use = int(expe_num % n_gpu)

            command = general_command.format(env_file, env_ext_file, model_config, model_ext, device_to_use, seed)
            processes.append(Popen(command, shell=True, env=os.environ.copy(), stderr=STDOUT))

for expe in processes:
    expe.wait()
print('Done running experiments')

for env_str, env_ext_str in product(env_config, env_ext):
    out_dir = "out/" + env_str + '_' + env_ext_str
    parse_env_subfolder(out_dir=out_dir)
