import matplotlib
matplotlib.use('Agg')
import json

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')

import os
import argparse
import numpy as np


# This is the default if success_threshold not defined in config
success_threshold_default = 0.65


def time_to_success(averaged_curve, success_threshold=success_threshold_default):
    tmp = np.where(averaged_curve > success_threshold)
    if tmp[0].shape == (0,):
        return len(averaged_curve)
    else:
        return tmp[0][0]

def plot_selected(env_dir, selected_list, name_spec=''):

    all_model_reward_mean_per_ep = []
    all_model_length_mean_per_ep = []

    for model_id in selected_list:

        mean_reward_file_path = os.path.join(env_dir, model_id, "mean_rewards_per_episode_stacked.npy")
        mean_length_file_path = os.path.join(env_dir, model_id, "mean_lengths_per_episode_stacked.npy")
        model_name = open(os.path.join(env_dir, model_id, "model_name"), 'r').read()

        all_model_reward_mean_per_ep.append( (np.load(mean_reward_file_path), model_name))
        all_model_length_mean_per_ep.append( (np.load(mean_length_file_path), model_name))


    # TODO : use averaged curve from aggregate_sub_folder_res
    palette = sns.color_palette(n_colors=len(all_model_length_mean_per_ep))

    plt.figure()
    for model_num, (reward_mean_per_ep, model_name) in enumerate(all_model_reward_mean_per_ep):
        sns.tsplot(data=reward_mean_per_ep, condition=model_name, color=palette[model_num])

    plt.savefig(os.path.join(env_dir, "model_curve_reward_summary{}.png".format(name_spec)))
    plt.close()

    for model_num, (length_mean_per_ep, model_name) in enumerate(all_model_length_mean_per_ep):
        sns.tsplot(data=length_mean_per_ep, condition=model_name, color=palette[model_num])

    plt.savefig(os.path.join(env_dir, "model_curve_length_summary{}.png".format(name_spec)))
    plt.close()

def plot_best_per_model(env_dir, num_model_taken = 3):

    best_resnet_dqn_model = []
    best_film_dqn_model = []

    summary_path = os.path.join(env_dir, 'summary')
    with open(summary_path) as summary_file:
        for line_num, line in enumerate(summary_file.readlines()):
            if len(best_resnet_dqn_model)>= num_model_taken and len(best_film_dqn_model)>=num_model_taken:
                break

            line_splitted = line.split(' ')
            name = line_splitted[0]
            print(line_splitted)
            model_id = line_splitted[1]

            if 'resnet_dqn' in name and len(best_resnet_dqn_model)< num_model_taken:
                best_resnet_dqn_model.append(model_id)

            elif 'dqn_filmed' in name and len(best_film_dqn_model) < num_model_taken:
                best_film_dqn_model.append(model_id)


    best_resnet_dqn_model.extend(best_film_dqn_model)
    plot_selected(env_dir, best_resnet_dqn_model, name_spec="_best_per_model")


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

            list_five_best_id.append(model_id)


    plot_selected(env_dir, list_five_best_id, name_spec="best{}".format(num_taken))


def parse_env_subfolder(out_dir):
    results = []
    results_new_obj = []

    for subfolder in os.listdir(out_dir):
        result_path = os.path.join(out_dir, (subfolder))

        if os.path.isfile(result_path):
            continue

        result_path += '/'

        results_sub = aggregate_sub_folder_res(result_path)

        if results_sub is None:
            continue

        name = results_sub['model_name']

        # Summary for main experiment
        mean_mean_reward = results_sub['mean_mean_reward']
        mean_mean_length = results_sub['mean_mean_length']

        time_to_success = results_sub['time_to_success']
        n_succeeded_train_objs = results_sub['number_of_succeeded_train_objs']
        results.append((name, subfolder, mean_mean_length, mean_mean_reward, time_to_success, n_succeeded_train_objs))

        # Summary for new_obj
        try:
            mean_mean_reward_new_obj = results_sub['mean_mean_reward_new_obj']
            mean_mean_length_new_obj = results_sub['mean_mean_length_new_obj']
            time_to_success_new_obj = results_sub['time_to_success_new_obj']
            n_succeeded_new_objs = results_sub['number_of_succeeded_new_objs']
        except KeyError:
            # If no test objective were present (max objective possible)
            mean_mean_reward_new_obj, mean_mean_length_new_obj, time_to_success_new_obj, n_succeeded_new_objs = -1, -1, -1, -1

        results_new_obj.append((name, subfolder, mean_mean_length_new_obj,
            mean_mean_reward_new_obj, time_to_success_new_obj, n_succeeded_new_objs))

    results.sort(key=lambda x:x[3])
    results_new_obj.sort(key=lambda x:x[3])


    print(results)
    print(results_new_obj)

    summary_str = ''
    for name, subfolder, length, reward, time, n_succ in results:
        summary_str += "{} {} {} {} {} {}\n".format(name, subfolder, length, reward, time, n_succ)

    summary_str_new_obj = ''
    for name, subfolder, length, reward, time, n_succ in results_new_obj:
        summary_str_new_obj += "{} {} {} {} {} {}\n".format(name, subfolder, length, reward, time, n_succ)

    open(out_dir+"/summary", 'w').write(summary_str)
    open(out_dir+"/summary_new_obj", 'w').write(summary_str_new_obj)

def aggregate_sub_folder_res(subfolder_path):
    # config_files.txt  config.json  eval_curve.png  last_10_std_length  last_10_std_reward  last_5_length.npy  last_5_reward.npy
    # length.npy  mean_length  mean_reward  model_name  reward.npy  train_lengths  train.log  train_rewards
    try:
        success_threshold = json.load(open(subfolder_path+"config.json", 'r'))['success_threshold']
        print('Using config defined threshold {}'.format(success_threshold))
    except KeyError:
        success_threshold = success_threshold_default
        print("Success threshold not defined in config, using default {}".format(success_threshold))

    results = dict()
    results['model_name'] = open(subfolder_path+"model_name", 'r').read()

    results['mean_mean_reward'] = 0
    results['mean_mean_length'] = 0

    results['mean_lengths_per_episode'] = []
    results['mean_rewards_per_episode'] = []

    results['mean_lengths_new_obj'] = []
    results['mean_rewards_new_obj'] = []

    results['mean_final_rewards_per_new_obj'] = []
    results['mean_final_rewards_per_obj'] = []

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

        #### PER OBJECTIVE STATS AGGREGATOR #######
        #==========================================
        # The outer loop averages over seeds, need to first average over objs
        per_obj_dir = seed_dir + 'per_obj/'
        try:
            n_objs = len(np.unique([int(i.split('_')[0][3:]) for i in os.listdir(per_obj_dir)]))
            for obj in range(n_objs):
                if obj == 0:
                    length_aggregator = np.loadtxt(per_obj_dir+"obj{}_lengths.txt".format(obj))
                    reward_aggregator = np.loadtxt(per_obj_dir+"obj{}_rewards.txt".format(obj))
                else:
                    length_aggregator = np.vstack((length_aggregator, np.loadtxt(per_obj_dir+"obj{}_lengths.txt".format(obj))))
                    reward_aggregator = np.vstack((reward_aggregator, np.loadtxt(per_obj_dir+"obj{}_rewards.txt".format(obj))))

            # For each obj, use last 5 tests to determine final exit time
            # Will be used for "Number of succeeded train objs"
            results['mean_final_rewards_per_obj'].append(np.mean(reward_aggregator[:, -5:], axis=1).flatten())

        except FileNotFoundError:
            # "No per_obj directory found, if experiments before introducing per_obj/ this is normal")
            pass


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

        # For each new obj, use last 5 tests to determine final exit time
        # Will be used for "Number of succeeded new objs"

        results['mean_final_rewards_per_new_obj'].append(np.mean(reward_aggregator[:, -5:], axis=1).flatten())


    ###### LEARNING : STATS'N PLOTS #####
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

    try:
        avg_rew_per_obj = np.stack(results['mean_final_rewards_per_obj'], axis=0).mean(axis=0)
        results['number_of_succeeded_train_objs'] = np.sum(avg_rew_per_obj > success_threshold)
    except:
        results['number_of_succeeded_train_objs'] = np.nan
    results['time_to_success'] = test_every * time_to_success(results['mean_rewards_per_episode'], success_threshold)

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

    ###### NEW OBJ : STATS'N PLOTS #####
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

        results['mean_lengths_new_obj'] = results['mean_lengths_new_obj_stacked'].mean(axis=0)
        results['mean_rewards_new_obj'] = results['mean_rewards_new_obj_stacked'].mean(axis=0)

        results['mean_mean_reward_new_obj'] = results['mean_lengths_new_obj'].mean()
        results['mean_mean_length_new_obj'] = results['mean_rewards_new_obj'].mean()

        results['std_lengths_new_obj'] = results['mean_lengths_new_obj_stacked'].std(axis=0)
        results['std_rewards_new_obj'] = results['mean_rewards_new_obj_stacked'].std(axis=0)

        results['std_lengths_new_obj'] = results['mean_lengths_new_obj_stacked'].std(axis=0)
        results['std_rewards_new_obj'] = results['mean_rewards_new_obj_stacked'].std(axis=0)

        # Add time to reach success_threshold as part of the results
        results['time_to_success_new_obj'] = 2 * test_every / n_objs * time_to_success(results['mean_rewards_new_obj'], success_threshold)

        # Add number of achieved new objs
        avg_rew_per_obj = np.stack(results['mean_final_rewards_per_new_obj'], axis=0).mean(axis=0)
        results['number_of_succeeded_new_objs'] = np.sum(avg_rew_per_obj > success_threshold)

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

    parser.add_argument("-out_dir", type=str, default='', help="Environment result directory (ex : out/multi_obj_test10every2")
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
            plot_best_per_model(env_dir=env_path)
    else:
        parse_env_subfolder(out_dir=out_dir)
        try:
            plot_best(env_dir=out_dir)
        except:
            print('Error encountered during plot_best')
        plot_best_per_model(env_dir=out_dir)
