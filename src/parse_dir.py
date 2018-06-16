import matplotlib
matplotlib.use('Agg')
import json

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')

import os
import argparse
import numpy as np

def plot_best(env_dir, num_taken=5):

    list_five_best_id = []
    with open(env_dir+"summary") as summary_file:
        for line_num, line in enumerate(summary_file.readline()):
            if line_num >= num_taken:
                break

            line_splitted = line.split(' ')
            name = line_splitted[0]
            print(line_splitted)
            model_id = line_splitted[1]

            list_five_best_id.append( (name,model_id) )

    all_model_reward_mean_per_ep = []
    all_model_length_mean_per_ep = []

    new_objs_reward_mean_per_ep = []
    new_objs_length_mean_per_ep = []

    for model_id in list_five_best_id:
        all_model_reward_mean_per_ep.append(np.load(env_dir + model_id + "mean_rewards_per_episode_stacked"))
        all_model_length_mean_per_ep.append(np.load(env_dir + model_id + "mean_lengths_per_episode_stacked"))

    # TODO : use averaged curve from aggregate_sub_folder_res

    plt.figure()
    for reward_mean_per_ep in all_model_reward_mean_per_ep:
        sns.tsplot(data=reward_mean_per_ep)

    plt.savefig(env_dir+"model_curve_summary.png")
    plt.close()

    for length_mean_per_ep in all_model_length_mean_per_ep:
        sns.tsplot(data=length_mean_per_ep)

    plt.savefig(env_dir + "model_curve_summary.png")
    plt.close()


def parse_env_subfolder(out_dir):
    results = []
    results_new_obj = []

    for subfolder in os.listdir(out_dir):
        result_path = out_dir + '/' + subfolder

        if os.path.isfile(result_path):
            continue

        result_path += '/'

        results_sub = aggregate_sub_folder_res(result_path)
        name = results_sub['model_name']

        # Summary for main experiment
        mean_mean_reward = results_sub['mean_mean_reward']
        mean_mean_length = results_sub['mean_mean_length']
        time_to_success = results_sub['time_to_success']
        results.append((name, subfolder, mean_mean_length, mean_mean_reward, time_to_success))

        # Summary for new_obj
        mean_mean_reward_new_obj = results_sub['mean_mean_reward_new_obj']
        mean_mean_length_new_obj = results_sub['mean_mean_length_new_obj']
        time_to_success_new_obj = results_sub['time_to_success_new_obj']
        results_new_obj.append((name, subfolder, mean_mean_length_new_obj, mean_mean_reward_new_obj, time_to_success_new_obj))



    results.sort(key=lambda x:x[2])
    results_new_obj.sort(key=lambda x:x[2])
    print(results)
    print(results_new_obj)

    summary_str = ''
    for name, subfolder, length, reward, time in results:
        summary_str += "{} {} {} {} {}\n".format(name, subfolder, length, reward, time)

    summary_str_new_obj = ''
    for name, subfolder, length, reward, time in results_new_obj:
        summary_str_new_obj += "{} {} {} {} {}\n".format(name, subfolder, length, reward, time)

    open(out_dir+"/summary", 'w').write(summary_str)
    open(out_dir+"/summary_new_obj", 'w').write(summary_str_new_obj)

def aggregate_sub_folder_res(subfolder_path):
    # config_files.txt  config.json  eval_curve.png  last_10_std_length  last_10_std_reward  last_5_length.npy  last_5_reward.npy
    # length.npy  mean_length  mean_reward  model_name  reward.npy  train_lengths  train.log  train_rewards
    results = dict()
    results['model_name'] = open(subfolder_path+"model_name", 'r').read()

    results['mean_mean_reward'] = 0
    results['mean_mean_length'] = 0

    results['mean_lengths_per_episode'] = []
    results['mean_rewards_per_episode'] = []

    results['mean_lengths_new_obj'] = []
    results['mean_rewards_new_obj'] = []

    n_different_seed = 0

    for file_in_subfolder in os.listdir(subfolder_path):
        seed_dir = subfolder_path + file_in_subfolder

        if os.path.isfile(seed_dir):
            continue

        seed_dir += '/'
        n_different_seed += 1

        results['mean_mean_reward'] += float(open(seed_dir+"mean_reward", 'r').read())
        results['mean_mean_length'] += float(open(seed_dir+"mean_length", 'r').read())

        results['mean_lengths_per_episode'].append(np.load(seed_dir+"length.npy"))
        results['mean_rewards_per_episode'].append(np.load(seed_dir+"reward.npy"))

        # The outer loop averages over seeds, need to first average over objs
        new_obj_dir = seed_dir + 'new_obj/'
        n_objs = len(np.unique([int(i.split('_')[0]) for i in os.listdir(new_obj_dir)]))

        for obj in range(n_objs):
            if obj == 0:
                length_aggregator = np.load(new_obj_dir+"{}_length.npy".format(obj))
                reward_aggregator = np.load(new_obj_dir+"{}_reward.npy".format(obj))
            else:
                length_aggregator = np.vstack((length_aggregator, np.load(new_obj_dir+"{}_length.npy".format(obj))))
                reward_aggregator = np.vstack((reward_aggregator, np.load(new_obj_dir+"{}_reward.npy".format(obj))))


        # This is averaged over objs
        results['mean_lengths_new_obj'].append(np.mean(length_aggregator, axis=0))
        results['mean_rewards_new_obj'].append(np.mean(reward_aggregator, axis=0))

    results['mean_mean_reward'] /= n_different_seed
    results['mean_mean_length'] /= n_different_seed

    results['mean_lengths_per_episode_stacked'] = np.stack(results['mean_lengths_per_episode'], axis=0)
    results['mean_rewards_per_episode_stacked'] = np.stack(results['mean_rewards_per_episode'], axis=0)

    results['mean_lengths_new_obj_stacked'] = np.stack(results['mean_lengths_new_obj'], axis=0)
    results['mean_rewards_new_obj_stacked'] = np.stack(results['mean_rewards_new_obj'], axis=0)

    results['mean_lengths_per_episode'] = results['mean_lengths_per_episode_stacked'].mean(axis=0)
    results['mean_rewards_per_episode'] = results['mean_rewards_per_episode_stacked'].mean(axis=0)

    results['mean_lengths_new_obj'] = results['mean_lengths_new_obj_stacked'].mean(axis=0)
    results['mean_rewards_new_obj'] = results['mean_rewards_new_obj_stacked'].mean(axis=0)

    results['mean_mean_reward_new_obj'] = results['mean_lengths_new_obj'].mean()
    results['mean_mean_length_new_obj'] = results['mean_rewards_new_obj'].mean()

    results['std_lengths_per_episode'] = results['mean_lengths_per_episode_stacked'].std(axis=0)
    results['std_rewards_per_episode'] = results['mean_rewards_per_episode_stacked'].std(axis=0)

    results['std_lengths_new_obj'] = results['mean_lengths_new_obj_stacked'].std(axis=0)
    results['std_rewards_new_obj'] = results['mean_rewards_new_obj_stacked'].std(axis=0)

    # Add time to reach success_threshold as part of the results
    def time_to_success(averaged_curve):
        success_threshold = 0.65
        tmp = np.where(averaged_curve > success_threshold)
        if tmp[0].shape == (0,):
            return len(averaged_curve)
        else:
            return tmp[0][0]


    # For plots and times to success, determine the scale
    test_every = json.load(open(subfolder_path + '/config.json', 'r'))["train_params"]["test_every"]
    n_objs = json.load(open(subfolder_path + '/config.json', 'r'))["env_type"]["objective"]["curriculum"]["n_objective"]

    most_objs_time = test_every * np.array(range(len(results['mean_lengths_per_episode_stacked'][0])))
    one_obj_time = 2 * test_every / n_objs * np.array(range(len(results['mean_lengths_new_obj_stacked'][0])))

    results['time_to_success'] = test_every * time_to_success(results['mean_rewards_per_episode'])
    results['time_to_success_new_obj'] = 2 * test_every / n_objs * time_to_success(results['mean_rewards_new_obj'])

    print(results['mean_lengths_per_episode_stacked'].shape, most_objs_time.shape)
    plt.figure()
    sns.tsplot(data=results['mean_lengths_per_episode_stacked'], time=most_objs_time)
    plt.savefig(subfolder_path+"mean_lengths_per_episode_over{}_run.png".format(n_different_seed))
    plt.close()

    plt.figure()
    sns.tsplot(data=results['mean_lengths_new_obj_stacked'], time=one_obj_time)
    plt.savefig(subfolder_path+"mean_lengths_new_obj_over{}_run.png".format(n_different_seed))
    plt.close()

    plt.figure()
    sns.tsplot(data=results['mean_rewards_per_episode_stacked'], time=most_objs_time)
    plt.savefig(subfolder_path+"mean_rewards_per_episode_over{}_run.png".format(n_different_seed))
    plt.close()

    plt.figure()
    sns.tsplot(data=results['mean_rewards_new_obj_stacked'], time=one_obj_time)
    plt.savefig(subfolder_path+"mean_rewards_new_obj_over{}_run.png".format(n_different_seed))
    plt.close()

    np.save(subfolder_path+"mean_rewards_per_episode_stacked", results['mean_rewards_per_episode_stacked'])
    np.save(subfolder_path+"mean_lengths_per_episode_stacked", results['mean_lengths_per_episode_stacked'])

    np.save(subfolder_path+"mean_rewards_new_obj_stacked", results['mean_rewards_new_obj_stacked'])
    np.save(subfolder_path+"mean_lengths_new_obj_stacked", results['mean_lengths_new_obj_stacked'])

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-out_dir", type=str, help="Directory with one expe")
    args = parser.parse_args()

    out_dir = args.out_dir
    parse_env_subfolder(out_dir=out_dir)
    try:
        plot_best(env_dir=out_dir)
    except:
        print('Error encountered during plot_best')
