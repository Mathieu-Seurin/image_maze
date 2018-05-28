import subprocess
from subprocess import Popen
from itertools import product
import os

from parse_dir import parse_env_subfolder, plot_best

test = False
verbose = True
if verbose:
    STDOUT = subprocess.STDOUT
else:
    STDOUT = subprocess.DEVNULL

env_folder = "config/env_base/"
env_ext_folder = "config/env_ext/"

model_folder = "config/model/"
model_ext_folder = "config/model_ext/"


#model_to_test = ['resnet_dqn_pretrain', 'dqn_filmed_pretrain', 'resnet_dqn', 'dqn_filmed']
model_to_test = ['resnet_dqn_pretrain', 'dqn_filmed_pretrain',]
extension_to_test = ['soft_update0_1', 'soft_update0_01', 'soft_update0_001', 'soft_update0_0001',
                      'hard_update0_1', 'hard_update0_01', 'hard_update0_001']

env_config = ["multi_obj_no_bkg", "multi_obj_class"]
env_ext = ["10obj_every2", "20obj_every2"]

n_gpu = 4
capacity_per_gpu = 4
n_seed = 5

if test:

    model_to_test = ['dqn_filmed']
    extension_to_test = ['soft_update0_01']

    env_config = ["multi_obj_test"]
    env_ext = ["10obj_every2", "20obj_every2"]

    n_gpu = 1
    n_seed = 10


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
while remains_command:

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
            processes[expe_num] = Popen(command, shell=True, env=os.environ.copy(), stderr=STDOUT)

for expe in processes:
    expe.wait()
print('Done running experiments')

for env_str, env_ext_str in product(env_config, env_ext):
    out_dir = "out/" + env_str + '_' + env_ext_str
    parse_env_subfolder(out_dir=out_dir)
    plot_best(env_dir=out_dir)