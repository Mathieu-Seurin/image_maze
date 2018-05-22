from subprocess import Popen
from config import write_seed_extensions
import os
import numpy as np
import matplotlib.pyplot as plt


def repeat_exp_parallel(task='', modifier='', agent='', agent_mod='', exp_dir='out', n_jobs=5):
    # Use 5 seeds to run 5 exps in parallel with results in subfolders

    # TODO : support agent modifier
    env_base = '../config/env_base/{}'.format(task)
    env_ext = '../config/env_ext/{}'.format(modifier)
    agent_base = '../config/model/{}'.format(agent)
    # agent_ext = '../config/model/{}'.format(agent_mod)

    exp_dir = '{}/{}/{}/{}'.format(exp_dir, task, modifier, agent)#, agent_mod)

    commands = [("python main.py -env_config {}.json -model_config {}.json "
                 "-env_extension {}.json -exp_dir {}/{} -seed {}").format(
                env_base, agent_base, env_ext, exp_dir, seed, seed)
                for seed in range(n_jobs)]
    processes = [Popen(cmd, shell=True) for cmd in commands]
    for p in processes:
        p.wait()
    print('Done running experiments')
    return exp_dir

def make_smoothed_training_curve(exp_dir='out/test_parallel'):
    # TODO : make it great again
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



for task in ["change_maze_10_full_black_and_white"]:
    for n_obj in [5, 10, 20]:
        modifier = '{}_every_1'.format(n_obj)
        for agent in ['reinforce_filmed_pretrain', 'reinforce_pretrain']:
            agent_cfg = '../config/{}'.format(agent)
            dir_ = repeat_exp_parallel(task, modifier, agent, exp_dir='out', n_jobs=3)
            # make_smoothed_training_curve(dir_)
