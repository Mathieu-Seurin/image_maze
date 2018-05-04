from subprocess import Popen
from config import write_seed_extensions
import os
import numpy as np
import matplotlib.pyplot as plt

# write_seed_extensions(range(10))

def repeat_exp_parallel(exp_dir='out/test_parallel'):
    # Use 5 seed extensions to run 5 exps in parallel with results in subfolders

    seed_extensions = ['../config/seed_extensions/{}'.format(i) for i in range(5)]
    commands = ['python main.py -config {}/cfg.json -extension {} -exp_dir {}'.format(
                exp_dir, ext, exp_dir) for ext in seed_extensions]

    processes = [Popen(cmd, shell=True) for cmd in commands]

    for p in processes:
        p.wait()
    print('Done running experiments')

def make_smoothed_training_curve(exp_dir='out/test_parallel'):
    x_, steps = [], []
    for subfolder in os.listdir(exp_dir):
        if subfolder in ['cfg.json', 'smoothed_training_curve.png']:
            continue
        tmp = np.loadtxt(exp_dir + '/' + subfolder + '/train_lengths')
        x_.append(tmp[:, 1])
        steps = tmp[:, 0]

    stack = np.array(x_[0])
    for x_list in x_[1:]:
        stack = np.vstack((stack, x_list))

    means, errors = np.mean(stack, axis=0), np.std(stack, axis=0)

    plt.figure()
    plt.errorbar(steps, means, errors)
    plt.savefig(exp_dir + '/smoothed_training_curve.png')
    plt.close()

# dir_ = 'out/test_reinforce'
# repeat_exp_parallel('../config/base_reinforce.json', out_dir=dir_)

# For now, need to manually put config file at root folder of parallel exp
# before launching it

for n_obj in [20]:
    for preprocessing in ['pretrained']:#, 'raw']:
        # dir_ = 'out/changing_obj_fixed_maze/{}_obj_every_10/reinforce/{}'.format(n_obj, preprocessing)
        dir_ = 'out/changing_obj_changing_maze/{}_obj_every_10/reinforce/{}'.format(n_obj, preprocessing)
        repeat_exp_parallel(exp_dir=dir_)
        make_smoothed_training_curve(dir_)
