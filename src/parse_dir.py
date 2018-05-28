import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')

import os
import argparse
import numpy as np

def plot_best(env_dir, num_taken=5):

    list_five_best_id = []
    summary_path = os.path.join(env_dir, 'summary')
    with open(summary_path) as summary_file:
        for line_num, line in enumerate(summary_file.readlines()):
            if line_num >= num_taken:
                break

            line_splitted = line.split(' ')
            name = line_splitted[0]
            model_id = line_splitted[1]

            list_five_best_id.append( (name,model_id) )

    all_model_reward_mean_per_ep = []
    all_model_length_mean_per_ep = []

    for name, model_id in list_five_best_id:
        mean_reward_file_path = os.path.join(env_dir, model_id, "mean_rewards_per_episode_stacked.npy")
        mean_length_file_path = os.path.join(env_dir, model_id, "mean_lengths_per_episode_stacked.npy")

        all_model_reward_mean_per_ep.append(np.load(mean_reward_file_path))
        all_model_length_mean_per_ep.append(np.load(mean_length_file_path))

    plt.figure()
    for reward_mean_per_ep in all_model_reward_mean_per_ep:
        sns.tsplot(data=reward_mean_per_ep)

    plt.savefig(env_dir+"model_curve_reward_summary.png")
    plt.close()

    for length_mean_per_ep in all_model_length_mean_per_ep:
        sns.tsplot(data=length_mean_per_ep)

    plt.savefig(env_dir + "model_curve_length_summary.png")
    plt.close()


def parse_env_subfolder(out_dir):
    results = []

    for subfolder in os.listdir(out_dir):
        result_path = os.path.join(out_dir, (subfolder))

        if os.path.isfile(result_path):
            continue

        result_path += '/'

        results_sub = aggregate_sub_folder_res(result_path)

        if results_sub is None:
            continue

        name = results_sub['model_name']
        mean_mean_reward = results_sub['mean_mean_reward']
        mean_mean_length = results_sub['mean_mean_length']

        results.append((name, subfolder, mean_mean_length, mean_mean_reward))

    results.sort(key=lambda x:x[2])
    print(results)

    summary_str = ''
    for name, subfolder, length, reward in results:
        summary_str += "{} {} {} {}\n".format(name, subfolder, length, reward)

    open(out_dir+"/summary", 'w').write(summary_str)

def aggregate_sub_folder_res(subfolder_path):
    # config_files.txt  config.json  eval_curve.png  last_10_std_length  last_10_std_reward  last_5_length.npy  last_5_reward.npy
    # length.npy  mean_length  mean_reward  model_name  reward.npy  train_lengths  train.log  train_rewards
    results = dict()

    results['model_name'] = open(subfolder_path+"model_name", 'r').read()

    results['mean_mean_reward'] = 0
    results['mean_mean_length'] = 0

    results['mean_lengths_per_episode'] = []
    results['mean_rewards_per_episode'] = []

    n_different_seed = 0

    for file_in_subfolder in os.listdir(subfolder_path):
        seed_dir = subfolder_path + file_in_subfolder

        if os.path.isfile(seed_dir):
            continue

        seed_dir += '/'
        n_different_seed += 1

        try :
            results['mean_mean_reward'] += float(open(seed_dir+"mean_reward", 'r').read())
        except FileNotFoundError :
            #print("Experiment failed : {}".format(seed_dir))
            continue
        results['mean_mean_length'] += float(open(seed_dir+"mean_length", 'r').read())

        results['mean_lengths_per_episode'].append(np.load(seed_dir+"length.npy"))
        results['mean_rewards_per_episode'].append(np.load(seed_dir+"reward.npy"))

    results['mean_mean_reward'] /= n_different_seed
    results['mean_mean_length'] /= n_different_seed

    try:
        results['mean_lengths_per_episode_stacked'] = np.stack(results['mean_lengths_per_episode'], axis=0)
    except ValueError:
        print(subfolder_path, " is an empty experiment")
        return None

    results['mean_rewards_per_episode_stacked'] = np.stack(results['mean_rewards_per_episode'], axis=0)

    results['mean_lengths_per_episode'] = results['mean_lengths_per_episode_stacked'].mean(axis=0)
    results['mean_rewards_per_episode'] = results['mean_rewards_per_episode_stacked'].mean(axis=0)

    results['std_lengths_per_episode'] = results['mean_lengths_per_episode_stacked'].std(axis=0)
    results['std_rewards_per_episode'] = results['mean_rewards_per_episode_stacked'].std(axis=0)

    plt.figure()
    sns.tsplot(data=results['mean_lengths_per_episode_stacked'])
    plt.savefig(subfolder_path+"mean_lengths_per_episode_over{}_run.png".format(n_different_seed))
    plt.close()

    plt.figure()
    sns.tsplot(data=results['mean_rewards_per_episode_stacked'])
    plt.savefig(subfolder_path+"mean_rewards_per_episode_over{}_run.png".format(n_different_seed))
    plt.close()

    np.save(subfolder_path+"mean_rewards_per_episode_stacked", results['mean_rewards_per_episode_stacked'])
    np.save(subfolder_path+"mean_lengths_per_episode_stacked", results['mean_lengths_per_episode_stacked'])

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-out_dir", type=str, default='' ,help="Environment result directory (ex : out/multi_obj_test10every2")
    args = parser.parse_args()

    out_dir = args.out_dir

    if out_dir == '':
        for out_dir in os.listdir('out/'):
            env_path = os.path.join('out', out_dir)
            if os.path.isfile(env_path):
                continue
            print("Parsing {}".format(env_path))
            print("=============================")
            parse_env_subfolder(out_dir=env_path)
            plot_best(env_dir=env_path)
    else:
        parse_env_subfolder(out_dir=out_dir)
        plot_best(env_dir=out_dir)




