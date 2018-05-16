from subprocess import Popen
from config import write_seed_extensions
import os
import numpy as np
import matplotlib.pyplot as plt


def repeat_exp_parallel(base_cfg='', agent_cfg='', exp_dir='out', n_jobs=5):
    # Use 5 seeds to run 5 exps in parallel with results in subfolders
    exp_dir = '{}/{}/{}'.format(exp_dir, base_cfg.split('/')[-1], agent_cfg.split('/')[-1])

    commands = ['python main.py -config {}.json -extension {}.json -exp_dir {}/{} -seed {}'.format(
                base_cfg, agent_cfg, exp_dir, seed, seed) for seed in range(n_jobs)]
    processes = [Popen(cmd, shell=True) for cmd in commands]
    for p in processes:
        p.wait()
    print('Done running experiments')
    return exp_dir

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




for n_obj in [3]:
    for preprocessing in ['preproc']:
        base_cfg = '../config/new_env/fix_maze_change_obj_{}_{}'.format(n_obj, preprocessing)
        for agent in ['baseline_dqn_filmed']:
            if preprocessing == 'preproc':
                agent = agent + '_preproc'
            agent_cfg = '../config/{}'.format(agent)
            dir_ = repeat_exp_parallel(base_cfg=base_cfg, agent_cfg=agent_cfg, exp_dir='out', n_jobs=5)
            make_smoothed_training_curve(dir_)
