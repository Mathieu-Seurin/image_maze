import matplotlib
matplotlib.use('Agg')
import json

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')

import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# This is the default if success_threshold not defined in config
success_threshold_default = 0.65


def time_to_success(averaged_curve, success_threshold=success_threshold_default):
    tmp = np.where(averaged_curve > success_threshold)
    if tmp[0].shape == (0,):
        return len(averaged_curve)
    else:
        return tmp[0][0]

def plot_selected(env_dir, selected_list, name_spec='', horizontal_scaling=False):
    all_model_reward_mean_per_ep = []
    all_model_length_mean_per_ep = []

    # Plot results for main train
    print("selected list in plot_selected", selected_list)
    test_every = json.load(open(os.path.join(env_dir, selected_list[0], "config.json"), 'r'))["train_params"]["test_every"]
    n_objs = json.load(open(os.path.join(env_dir, selected_list[0], "config.json"), 'r'))["env_type"]["objective"]["curriculum"]["n_objective"]

    if horizontal_scaling:
        most_objs_time = test_every * np.array(range(len(np.load(os.path.join(env_dir, selected_list[0], "mean_rewards_per_episode_stacked.npy"))[0])))
    else:
        most_objs_time = None

    for model_id in selected_list:

        mean_reward_file_path = os.path.join(env_dir, model_id, "mean_rewards_per_episode_stacked.npy")
        mean_length_file_path = os.path.join(env_dir, model_id, "mean_lengths_per_episode_stacked.npy")
        model_name = open(os.path.join(env_dir, model_id, "model_name"), 'r').read()

        all_model_reward_mean_per_ep.append( (np.load(mean_reward_file_path), model_name))
        all_model_length_mean_per_ep.append( (np.load(mean_length_file_path), model_name))


    palette = sns.color_palette(n_colors=len(all_model_length_mean_per_ep))

    plt.figure()
    for model_num, (reward_mean_per_ep, model_name) in enumerate(all_model_reward_mean_per_ep):
        if np.any(most_objs_time):
            if reward_mean_per_ep.shape[1] < len(most_objs_time):
                sns.tsplot(data=reward_mean_per_ep, condition=model_name, color=palette[model_num], time=most_objs_time[:reward_mean_per_ep.shape[1]])
            else:
                sns.tsplot(data=reward_mean_per_ep, condition=model_name, color=palette[model_num], time=most_objs_time)
        else:
            sns.tsplot(data=reward_mean_per_ep, condition=model_name, color=palette[model_num])

    plt.ylim(-0.05, 1.05)
    plt.savefig(os.path.join(env_dir, "model_curve_reward_summary{}.svg".format(name_spec)))
    plt.close()


    plt.figure()
    for model_num, (length_mean_per_ep, model_name) in enumerate(all_model_length_mean_per_ep):
        if np.any(most_objs_time):
            if length_mean_per_ep.shape[1] < len(most_objs_time):
                sns.tsplot(data=length_mean_per_ep, condition=model_name, color=palette[model_num], time=most_objs_time[:length_mean_per_ep.shape[1]])
            else:
                sns.tsplot(data=length_mean_per_ep, condition=model_name, color=palette[model_num], time=most_objs_time)
        else:
            sns.tsplot(data=length_mean_per_ep, condition=model_name, color=palette[model_num])


    plt.savefig(os.path.join(env_dir, "model_curve_length_summary{}.svg".format(name_spec)))
    plt.close()

    # Plots for new_obj
    # Assume that for a given model, best train performances imply best generalization


    try:
        if horizontal_scaling:
            one_obj_time =  test_every // 3 * np.array(range(len(np.load(os.path.join(env_dir, selected_list[0], "mean_rewards_new_obj_stacked.npy"))[0])))
            # This was for one obj at a time
            # one_obj_time = 2 * test_every / n_objs * np.array(range(len(np.load(os.path.join(env_dir, selected_list[0], "mean_rewards_new_obj_stacked.npy"))[0])))
        else:
            one_obj_time = None

        all_model_reward_mean_per_ep = []
        all_model_length_mean_per_ep = []
        for model_id in selected_list:
            mean_reward_file_path = os.path.join(env_dir, model_id, "mean_rewards_new_obj_stacked.npy")
            mean_length_file_path = os.path.join(env_dir, model_id, "mean_lengths_new_obj_stacked.npy")
            model_name = open(os.path.join(env_dir, model_id, "model_name"), 'r').read()

            all_model_reward_mean_per_ep.append( (np.load(mean_reward_file_path), model_name))
            all_model_length_mean_per_ep.append( (np.load(mean_length_file_path), model_name))

        palette = sns.color_palette(n_colors=len(all_model_length_mean_per_ep))

        plt.figure()
        for model_num, (length_mean_per_ep, model_name) in enumerate(all_model_length_mean_per_ep):
            if np.any(one_obj_time):
                if length_mean_per_ep.shape[1] < len(most_objs_time):
                    sns.tsplot(data=length_mean_per_ep, condition=model_name, color=palette[model_num], time=one_obj_time[:length_mean_per_ep.shape[1]])
                elif reward_mean_per_ep.shape[1] > len(most_objs_time):
                    sns.tsplot(data=length_mean_per_ep, condition=model_name, color=palette[model_num], time=one_obj_time)
                else:
                    sns.tsplot(data=length_mean_per_ep, condition=model_name, color=palette[model_num], time=one_obj_time)
            else:
                sns.tsplot(data=length_mean_per_ep, condition=model_name, color=palette[model_num])


        plt.savefig(os.path.join(env_dir, "new_obj_length_summary{}.svg".format(name_spec)))
        plt.close()

        plt.figure()

        for model_num, (reward_mean_per_ep, model_name) in enumerate(all_model_reward_mean_per_ep):
            if np.any(one_obj_time):
                if reward_mean_per_ep.shape[1] < len(most_objs_time):
                    sns.tsplot(data=reward_mean_per_ep, condition=model_name, color=palette[model_num], time=one_obj_time[:reward_mean_per_ep.shape[1]])
                else:
                    sns.tsplot(data=reward_mean_per_ep, condition=model_name, color=palette[model_num], time=one_obj_time)
            else:
                sns.tsplot(data=reward_mean_per_ep, condition=model_name, color=palette[model_num])
        plt.ylim(-0.05, 1.05)
        plt.savefig(os.path.join(env_dir, "new_obj_reward_summary{}.svg".format(name_spec)))
        plt.close()

    except FileNotFoundError:
        print('New obj results not found')



def plot_best_per_model(env_dir, num_model_taken=1):
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
    if best_resnet_dqn_model:
        plot_selected(env_dir, best_resnet_dqn_model, name_spec="_best_per_model_dqn", horizontal_scaling=True)

    all_selected_reinforce = []
    best_reinforce_model = []
    best_film_reinforce_model = []
    best_reinforce_pretrain_model = []
    best_film_reinforce_pretrain_model = []

    summary_path = os.path.join(env_dir, 'summary')
    print('\nBegin treating per model reinforce')
    with open(summary_path) as summary_file:
        for line_num, line in enumerate(summary_file.readlines()):
            # if len(best_resnet_dqn_model)>= num_model_taken and len(best_film_dqn_model)>=num_model_taken:
            #     break

            line_splitted = line.split(' ')
            name = line_splitted[0]
            print(line_splitted)
            model_id = line_splitted[1]

            if 'reinforce_filmed' in name:
                if 'pretrain' in name and len(best_film_reinforce_pretrain_model) < num_model_taken:
                    best_film_reinforce_pretrain_model.append(model_id)
                    print('plop', model_id)
                elif 'pretrain' not in name and len(best_film_reinforce_model) < num_model_taken:
                    best_film_reinforce_model.append(model_id)
                    print('plop2', model_id)

            elif 'reinforce' in name and not "filmed" in name:
                if 'pretrain' in name and len(best_reinforce_pretrain_model) < num_model_taken:
                    best_reinforce_pretrain_model.append(model_id)
                    print('plop3', model_id)
                elif 'pretrain' not in name and len(best_reinforce_model) < num_model_taken:
                    best_reinforce_model.append(model_id)
                    print('plop4', model_id)
    for model_list in [best_reinforce_model, best_film_reinforce_model, best_reinforce_pretrain_model, best_film_reinforce_pretrain_model]:
        all_selected_reinforce.extend(model_list)

    print("all_selected_reinforce", all_selected_reinforce)
    if all_selected_reinforce:
        plot_selected(env_dir, all_selected_reinforce, name_spec="_best_per_model_reinforce", horizontal_scaling=True)


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

    if list_five_best_id:
        plot_selected(env_dir, list_five_best_id, name_spec="best{}".format(num_taken))


def parse_env_subfolder(out_dir):
    results = []
    results_new_obj = []

    get_deterministic_agents_score(out_dir)

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

    results.sort(key=lambda x:-x[3])
    results_new_obj.sort(key=lambda x:-x[3])


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

def get_deterministic_agents_score(out_dir):
    rews_perf = []
    rews_rand = []

    for subfolder in os.listdir(out_dir):
        subfolder_path = os.path.join(out_dir, (subfolder))

        if os.path.isfile(subfolder_path):
            continue

        subfolder_path += '/'

        name = open(subfolder_path+"model_name", 'r').read()

        if name not in ['base_perfect', 'basic_config']:
            continue

        n_different_seed = 0

        for file_in_subfolder in os.listdir(subfolder_path):
            seed_dir = subfolder_path + file_in_subfolder

            if os.path.isfile(seed_dir):
                continue

            seed_dir += '/'
            n_different_seed += 1

            #### PER OBJECTIVE STATS AGGREGATOR #######
            #==========================================
            # The outer loop averages over seeds, need to first average over objs
            per_obj_dir = seed_dir + 'per_obj/'

            n_objs = len(np.unique([int(i.split('_')[0][3:]) for i in os.listdir(per_obj_dir)]))
            for obj in range(n_objs):
                if name == 'base_perfect':
                    rews_perf.append(np.loadtxt(per_obj_dir+"obj{}_rewards.txt".format(obj)))
                if name == 'basic_config':
                    rews_rand.append(np.loadtxt(per_obj_dir+"obj{}_rewards.txt".format(obj)))

    np.savetxt(os.path.join(out_dir, 'perfect_agent_rew.txt'), [np.mean(rews_perf)])
    np.savetxt(os.path.join(out_dir, 'random_agent_rew.txt'), [np.mean(rews_rand)])


def aggregate_sub_folder_res(subfolder_path):
    # config_files.txt  config.json  eval_curve.svg  last_10_std_length  last_10_std_reward  last_5_length.npy  last_5_reward.npy
    # length.npy  mean_length  mean_reward  model_name  reward.npy  train_lengths  train.log  train_rewards
    try:
        success_threshold = json.load(open(subfolder_path+"config.json", 'r'))['success_threshold']
        print('Using config defined threshold {}'.format(success_threshold))
    except KeyError:
        success_threshold = success_threshold_default
        print("Success threshold not defined in config, using default {}".format(success_threshold))

    results = dict()
    results['model_name'] = open(subfolder_path+"model_name", 'r').read()

    results['perfect_agent_rew'] = 1.
    results['random_agent_rew'] = 0.
    out_dir = subfolder_path + '../'

    try:
        results['perfect_agent_rew'] = np.loadtxt(os.path.join(out_dir, 'perfect_agent_rew.txt'))
        results['random_agent_rew'] = np.loadtxt(os.path.join(out_dir, 'random_agent_rew.txt'))
    except OSError:
        print("Did not find the perfect / random agent rewards at {}".format(os.path.join(out_dir, 'perfect_agent_rew.txt')))
        pass

    print(results['perfect_agent_rew'], results['random_agent_rew'])
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
                    try:
                        length_aggregator = np.vstack((length_aggregator, np.loadtxt(per_obj_dir+"obj{}_lengths.txt".format(obj))))
                        reward_aggregator = np.vstack((reward_aggregator, np.loadtxt(per_obj_dir+"obj{}_rewards.txt".format(obj))))
                    except ValueError:
                        pass
            # For each obj, use last 5 tests to determine final exit time
            # Will be used for "Number of succeeded train objs"
            results['mean_final_rewards_per_obj'].append(np.mean(reward_aggregator[:, -5:], axis=1).flatten())

        except FileNotFoundError:
            # "No per_obj directory found, if experiments before introducing per_obj/ this is normal")
            pass
        except ValueError:
            # This indicates that new_objs were done using test_bulk
            pass

        except IndexError:
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
        except ValueError:
            # This indicates that new_objs were done using test_bulk
            new_obj_dir = seed_dir + 'new_obj/' + 'per_obj/'
            n_objs = len(np.unique([int(i.split('_')[0][3:]) for i in os.listdir(new_obj_dir)]))
            for obj in range(n_objs):
                if obj == 0:
                    length_aggregator = np.loadtxt(new_obj_dir+"obj{}_lengths.txt".format(obj))
                    reward_aggregator = np.loadtxt(new_obj_dir+"obj{}_rewards.txt".format(obj))
                else:
                    try:
                        length_aggregator = np.vstack((length_aggregator, np.loadtxt(new_obj_dir+"obj{}_lengths.txt".format(obj))))
                        reward_aggregator = np.vstack((reward_aggregator, np.loadtxt(new_obj_dir+"obj{}_rewards.txt".format(obj))))
                    except ValueError:
                        print("Exp wasn't done, ignore it")
                        pass


            # This is averaged over objs
            results['mean_lengths_new_obj'].append(np.mean(length_aggregator, axis=0))
            results['mean_rewards_new_obj'].append(np.mean(reward_aggregator, axis=0))

            # For each new obj, use last 5 tests to determine final exit time
            # Will be used for "Number of succeeded new objs"

            results['mean_final_rewards_per_new_obj'].append(np.mean(reward_aggregator[:, -5:], axis=1).flatten())
            continue

        for obj in range(n_objs):
            if obj == 0:
                length_aggregator = np.load(new_obj_dir+"{}_length.npy".format(obj))
                reward_aggregator = np.load(new_obj_dir+"{}_reward.npy".format(obj))
            else:
                try:
                    length_aggregator = np.vstack((length_aggregator, np.load(new_obj_dir+"{}_length.npy".format(obj))))
                    reward_aggregator = np.vstack((reward_aggregator, np.load(new_obj_dir+"{}_reward.npy".format(obj))))
                except:
                    print('Error line 360 or 361 : most likely one of the exps crashed')
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

    # Account for crashed experiments :
    max_len = np.max([len(i) for i in results['mean_lengths_per_episode']])
    results['mean_lengths_per_episode'] = [l for l in results['mean_lengths_per_episode'] if len(l) == max_len]
    results['mean_rewards_per_episode'] = [l for l in results['mean_rewards_per_episode'] if len(l) == max_len]

    max_len_new = np.max([len(i) for i in results['mean_lengths_new_obj']])
    results['mean_lengths_new_obj'] = [l for l in results['mean_lengths_new_obj'] if len(l) == max_len_new]
    results['mean_rewards_new_obj'] = [l for l in results['mean_rewards_new_obj'] if len(l) == max_len_new]

    # For plots, determine the scale
    test_every = json.load(open(subfolder_path + '/config.json', 'r'))["train_params"]["test_every"]
    n_objs = json.load(open(subfolder_path + '/config.json', 'r'))["env_type"]["objective"]["curriculum"]["n_objective"]

    most_objs_time = test_every * np.array(range(max_len))

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
    plt.savefig(os.path.join(subfolder_path, "mean_lengths_per_episode_over{}_run.svg".format(n_different_seed)))
    plt.close()

    # Scale the rewards before saving them (that way, will automatically be used in the following)

    results['mean_rewards_per_episode_stacked'] -= results['random_agent_rew']
    results['mean_rewards_per_episode_stacked'] /= (results['perfect_agent_rew'] - results['random_agent_rew'])

    plt.figure()
    sns.tsplot(data=results['mean_rewards_per_episode_stacked'], time=most_objs_time)
    plt.savefig(os.path.join(subfolder_path, "mean_rewards_per_episode_over{}_run.svg".format(n_different_seed)))
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

        results['mean_mean_reward_new_obj'] = results['mean_rewards_new_obj'].mean()
        results['mean_mean_length_new_obj'] = results['mean_lengths_new_obj'].mean()

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
        plt.savefig(os.path.join(subfolder_path,"mean_lengths_new_obj_over{}_run.svg".format(n_different_seed)))
        plt.close()

        results['mean_rewards_new_obj_stacked'] -= results['random_agent_rew']
        results['mean_rewards_new_obj_stacked'] /= (results['perfect_agent_rew'] - results['random_agent_rew'])

        plt.figure()
        sns.tsplot(data=results['mean_rewards_new_obj_stacked'], time=one_obj_time)
        plt.savefig(os.path.join(subfolder_path,"mean_rewards_new_obj_over{}_run.svg".format(n_different_seed)))
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
            if os.path.isfile(env_path) or out_dir[0] == "_": # "_" means ignore this folder, to keep result and speed up process
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
