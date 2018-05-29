import matplotlib
matplotlib.use('Agg')
import json

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')

import os
import argparse
import numpy as np

def plot_selected(env_dir, selected_list):
    pass

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


    # TODO : use averaged curve from aggregate_sub_folder_res

    plt.figure()
    for reward_mean_per_ep in all_model_reward_mean_per_ep:
        sns.tsplot(data=reward_mean_per_ep)

    plt.savefig(os.path.join(env_dir, "model_curve_reward_summary.png"))
    plt.close()

    for length_mean_per_ep in all_model_length_mean_per_ep:
        sns.tsplot(data=length_mean_per_ep)

    plt.savefig(os.path.join(env_dir, "model_curve_length_summary.png"))
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

    results['mean_lengths_new_obj'] = []
    results['mean_rewards_new_obj'] = []

    n_different_seed = 0

    for file_in_subfolder in os.listdir(subfolder_path):
        seed_dir = subfolder_path + file_in_subfolder

        if os.path.isfile(seed_dir):
            continue

        seed_dir += '/'
        n_different_seed += 1

        ###### LEARNING STATS AGGREGATOR #####
        #====================================

        try :
            results['mean_mean_reward'] += float(open(seed_dir+"mean_reward", 'r').read())
        except FileNotFoundError :
            #print("Experiment failed : {}".format(seed_dir))
            continue
        results['mean_mean_length'] += float(open(seed_dir+"mean_length", 'r').read())

        results['mean_lengths_per_episode'].append(np.load(seed_dir+"length.npy"))
        results['mean_rewards_per_episode'].append(np.load(seed_dir+"reward.npy"))


        #### NEW OBJECTIVES STATS AGGREGATOR #######
        #==========================================
        # The outer loop averages over seeds, need to first average over objs
        new_obj_dir = seed_dir + 'new_obj/'
        try:
            n_objs = len(np.unique([int(i.split('_')[0]) for i in os.listdir(new_obj_dir)]))
        except FileNotFoundError:
            # "No new_obj directory found, if there was 20 objectives during training, this is normal")
            continue

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


    ###### LEARNING STATS'N PLOTS #####
    #==================================

    results['mean_mean_reward'] /= n_different_seed
    results['mean_mean_length'] /= n_different_seed

    try:
        results['mean_lengths_per_episode_stacked'] = np.stack(results['mean_lengths_per_episode'], axis=0)
    except ValueError:
        print(subfolder_path, " is an empty experiment")
        return None

    # For plots, determine the scale
    test_every = json.load(open(subfolder_path + '/config.json', 'r'))["train_params"]["test_every"]
    n_objs = json.load(open(subfolder_path + '/config.json', 'r'))["env_type"]["objective"]["curriculum"]["n_objective"]

    most_objs_time = test_every * np.array(range(len(results['mean_lengths_per_episode_stacked'][0])))

    results['mean_rewards_per_episode_stacked'] = np.stack(results['mean_rewards_per_episode'], axis=0)

    results['mean_lengths_per_episode'] = results['mean_lengths_per_episode_stacked'].mean(axis=0)
    results['mean_rewards_per_episode'] = results['mean_rewards_per_episode_stacked'].mean(axis=0)

    results['std_lengths_per_episode'] = results['mean_lengths_per_episode_stacked'].std(axis=0)
    results['std_rewards_per_episode'] = results['mean_rewards_per_episode_stacked'].std(axis=0)

    # print(results['mean_lengths_per_episode_stacked'].shape, most_objs_time.shape)
    plt.figure()
    sns.tsplot(data=results['mean_lengths_per_episode_stacked'], time=most_objs_time)
    plt.savefig(os.path.join(subfolder_path, "mean_lengths_per_episode_over{}_run.png".format(n_different_seed)))
    plt.close()

    plt.figure()
    sns.tsplot(data=results['mean_rewards_per_episode_stacked'], time=most_objs_time)
    plt.savefig(os.path.join(subfolder_path, "mean_rewards_per_episode_over{}_run.png".format(n_different_seed)))
    plt.close()

    np.save(os.path.join(subfolder_path, "mean_rewards_per_episode_stacked"), results['mean_rewards_per_episode_stacked'])
    np.save(os.path.join(subfolder_path, "mean_lengths_per_episode_stacked"), results['mean_lengths_per_episode_stacked'])

    ###### NEW OBJ STATS'N PLOTS #####
    #=================================

    # If you have 20 objectives, no new obj possible.
    try:
        results['mean_lengths_new_obj_stacked'] = np.stack(results['mean_lengths_new_obj'], axis=0)
        new_obj_test_is_available = True
    except ValueError:
        new_obj_test_is_available = False
        print("No new_obj directory found, if there was 20 objectives during training, this is normal")

    if new_obj_test_is_available:
        one_obj_time = 2 * test_every / n_objs * np.array(range(len(results['mean_lengths_new_obj_stacked'][0])))

        results['mean_lengths_new_obj_stacked'] = np.stack(results['mean_lengths_new_obj'], axis=0)
        results['mean_rewards_new_obj_stacked'] = np.stack(results['mean_rewards_new_obj'], axis=0)

        results['std_lengths_new_obj'] = results['mean_lengths_new_obj_stacked'].std(axis=0)
        results['std_rewards_new_obj'] = results['mean_rewards_new_obj_stacked'].std(axis=0)

        plt.figure()
        sns.tsplot(data=results['mean_lengths_new_obj_stacked'], time=one_obj_time)
        plt.savefig(os.path.join(subfolder_path,"mean_lengths_new_obj_over{}_run.png".format(n_different_seed)))
        plt.close()


        plt.figure()
        sns.tsplot(data=results['mean_rewards_new_obj_stacked'], time=one_obj_time)
        plt.savefig(os.path.join(subfolder_path,"mean_rewards_new_obj_over{}_run.png".format(n_different_seed)))
        plt.close()

        np.save(os.path.join(subfolder_path,"mean_rewards_new_obj_stacked"), results['mean_rewards_new_obj_stacked'])
        np.save(os.path.join(subfolder_path,"mean_lengths_new_obj_stacked"), results['mean_lengths_new_obj_stacked'])

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