from subprocess import Popen
from config import write_seed_extensions
import os
import numpy as np
import matplotlib.pyplot as plt

# write_seed_extensions(range(5))

def repeat_exp_parallel(base_cfg_name, out_dir='out/test_parallel'):
    # Use the 5 seed extensions to run 5 exps in parallel with results in the
    # subfolders of the same folder
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    seed_extensions = ['../config/seed_extensions/{}'.format(i) for i in range(5)]

    commands = ['python main.py -config {} -extension {} -exp_dir {}'.format(
                base_cfg_name, ext, out_dir) for ext in seed_extensions]

    processes = [Popen(cmd, shell=True) for cmd in commands]

    for p in processes:
        p.wait()

def make_smoothed_training_curve(exp_dir='out/test_parallel'):
    x_, steps = [], []
    for subfolder in os.listdir(exp_dir):
        tmp = np.loadtxt(exp_dir + '/' + subfolder + '/train_lengths')
        x_.append(tmp[:, 1])
        steps = tmp[:, 0]

    stack = np.array(x_[0])
    for x_list in x_[1:]:
        stack = np.vstack((stack, x_list))

    means, errors = np.mean(stack, axis=0), np.std(stack, axis=0)

    plt.errorbar(steps, means, errors)
    plt.savefig(exp_dir + '/smoothed_training_curve.png')

dir = 'out/task2_10_objs_reinforce'
repeat_exp_parallel('../config/base_reinforce.json', out_dir=dir)
make_smoothed_training_curve(dir)

# T5 : 10 objectives random no bkg
