import subprocess
from subprocess import Popen
from itertools import product
import os

from parse_dir import parse_env_subfolder

env_config = "change_maze_10_random_image_no_bkg"
env_ext = "10_every_1"

verbose = True
if verbose:
    STDOUT = subprocess.STDOUT
else:
    STDOUT = subprocess.DEVNULL

env_folder = "config/env_base/"
env_ext_folder = "config/env_ext/"

model_to_test = ['dqn_filmed_pretrain']
extension_to_test = ['soft_update0_0001',]

model_folder = "config/model/"
model_ext_folder = "config/model_ext/"

n_gpu = 1
capacity_per_gpu = 3
n_seed = 5

seeds = [i for i in range(n_seed)]

general_command = "python3 src/main.py -env_config {}.json -env_extension {}.json -model_config {}.json -model_extension {}.json -device {} -seed {}"

#Fill queue with all your expe
command_keys = ['model', 'model_ext', 'seed']
all_commands = [dict(zip(command_keys, command_param)) for command_param in product(model_to_test, extension_to_test, seeds)]
print("{} experiments to run.".format(len(all_commands)))
processes = []

# Launch expe, fill all processes
for expe_num in range(n_gpu*capacity_per_gpu):
    try:
        command_param = all_commands.pop()
    except IndexError:
        break

    model_config = model_folder+command_param['model']
    model_ext = model_ext_folder+command_param['model_ext']
    seed = command_param['seed']
    device_to_use = int(expe_num%n_gpu)

    command = general_command.format(env_folder+env_config, env_ext_folder+env_ext, model_config, model_ext, device_to_use, seed)
    processes.append(Popen(command, shell=True, env=os.environ.copy(), stderr=STDOUT))

print("{} expes remains at the moment".format(len(all_commands)))
remains_command = len(all_commands) > 0
while remains_command:

    for expe_num, expe in enumerate(processes):
        if expe.poll() is not None:
            try:
                command_param = all_commands.pop()
                print("Launching new expe, {} expe remains".format(len(all_commands)))

            except IndexError:
                remains_command = False
                break

            model_config = model_folder + command_param['model']
            model_ext = model_ext_folder + command_param['model_ext']
            seed = command_param['seed']
            device_to_use = int(expe_num % n_gpu)

            command = general_command.format(env_folder + env_config, env_ext_folder + env_ext, model_config, model_ext,
                                             device_to_use, seed)
            processes[expe_num] = Popen(command, shell=True, env=os.environ.copy(), stderr=STDOUT)

for expe in processes:
    expe.wait()
print('Done running experiments')

out_dir = "out/" + env_config + '_' + env_ext
parse_env_subfolder(out_dir=out_dir)
