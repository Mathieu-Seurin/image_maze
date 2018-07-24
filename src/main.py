import matplotlib
matplotlib.use('Agg')

import argparse
import logging
import time
from itertools import count

import numpy as np
from config import load_config_and_logger, set_seed, save_stats, compute_epsilon_schedule
from image_text_utils import make_video, make_eval_plot
import os
import sys
from copy import deepcopy
import torch

def full_train_test(env_config, model_config, env_extension, model_extension, exp_dir, seed=0, device=-1, args=None):

    # Load_config also creates logger inside (INFO to stdout, INFO to train.log)
    config, exp_identifier, save_path = load_config_and_logger(env_config_file=env_config,
                                                               model_config_file=model_config,
                                                               env_ext_file=env_extension,
                                                               model_ext_file=model_extension,
                                                               args=args,
                                                               exp_dir=exp_dir,
                                                               seed=seed
                                                               )

    logger = logging.getLogger()

    if device != -1:
        torch.cuda.set_device(device)
        logger.info("Using device {}".format(torch.cuda.current_device()))
    else:
        logger.info("Using default device from env")

    from feature_maze import ImageFmapGridWorld
    from rl_agent.basic_agent import AbstractAgent, PerfectAgent
    from rl_agent.dqn_agent import DQNAgent
    from rl_agent.reinforce_agent import ReinforceAgent

    verbosity = config["io"]["verbosity"]
    save_image = bool(config["io"]["num_epochs_to_store"])

    env = ImageFmapGridWorld(config=config["env_type"], features_type=config["images_features"], save_image=save_image)

    if config["agent_type"] == 'random':
        rl_agent = AbstractAgent(config, env.action_space())
        discount_factor = 0.90
        return test(agent=rl_agent,
                    env=env,
                    config=config,
                    test_number=1,
                    discount_factor=discount_factor,
                    save_path=save_path)

    elif 'dqn' in config["agent_type"]:
        rl_agent = DQNAgent(config, env.action_space(), env.state_objective_dim(), env.is_multi_objective, env.objective_type)
        discount_factor = config["resnet_dqn_params"]["discount_factor"]

    elif 'reinforce' in config["agent_type"]:
        rl_agent = ReinforceAgent(config, env.action_space(), env.state_objective_dim(), env.is_multi_objective, env.objective_type)
        discount_factor = config["resnet_reinforce_params"]["discount_factor"]

    elif config['agent_type'] == 'perfect':
        rl_agent = PerfectAgent(config, env.action_space())
        discount_factor = config["discount_factor"]
        return test(agent=rl_agent,
                    env=env,
                    config=config,
                    test_number=1,
                    discount_factor=discount_factor,
                    save_path=save_path)

    else:
        assert False, "Wrong agent type : {}".format(config["agent_type"])

    n_epochs = config["train_params"]["n_epochs"]
    test_every = config["train_params"]["test_every"]
    eps_range = compute_epsilon_schedule(config['train_params'], n_epochs)

    do_zero_shot_test = False
    do_new_obj_dynamics = True

    tic = time.time()
    logging.info(" ")
    logging.info("Begin Training")
    logging.info("===============")

    reward_list = []
    length_list = []


    for epoch in range(n_epochs):
        state = env.reset(show=False)
        done = False
        time_out = 20
        num_step = 0

        if epoch % test_every == 0:
            reward, length = test(agent=rl_agent,
                                  env=env,
                                  config=config,
                                  test_number=epoch,
                                  discount_factor=discount_factor,
                                  save_path=save_path)

            logging.info("Epoch {} test : averaged reward {:.2f}, average length {:.2f}".format(epoch, reward, length))

            with open(save_path.format('train_lengths'), 'a+') as f:
                f.write("{} {}\n".format(epoch, length))
                length_list.append(length)
            with open(save_path.format('train_rewards'), 'a+') as f:
                f.write("{} {}\n".format(epoch, reward))
                reward_list.append(reward)
            make_eval_plot(save_path.format('train_lengths'), save_path.format('eval_curve.png'))
            make_eval_plot(save_path.format('train_rewards'), save_path.format('eval_curve_rew.png'))
            if do_zero_shot_test :
                reward, length = test_zero_shot(agent=rl_agent,
                                                env=env,
                                                config=config,
                                                test_number=epoch,
                                                save_path=save_path,
                                                discount_factor=discount_factor)

                logging.info("Epoch {} zero-shot test : averaged reward {:.2f}, average length {:.2f}".format(epoch, reward, length))

                with open(save_path.format('zero_shot_lengths'), 'a+') as f:
                    f.write("{} {}\n".format(epoch, length))
                    length_list.append(length)
                with open(save_path.format('zero_shot_rewards'), 'a+') as f:
                    f.write("{} {}\n".format(epoch, reward))
                    reward_list.append(reward)
                make_eval_plot(save_path.format('zero_shot_lengths'), save_path.format('eval_curve.png'))

        while not done and num_step < time_out:
            num_step += 1
            action = rl_agent.forward(state, eps_range[epoch])
            next_state, reward, done, _info = env.step(action)

            assert bool(reward) == bool(done), "Holy shit."
            assert bool(reward) == bool(_info['agent_position'] == _info['reward_position']), "Epic Fail"

            loss = rl_agent.optimize(state, action, next_state, reward)
            state = next_state

        rl_agent.callback(epoch)
        env.post_process()

    save_stats(save_path, reward_list, length_list)
    toc = time.time()
    logging.info('Total time for main obj loop : {}'.format(toc - tic))

    if do_new_obj_dynamics:
        try:
            os.makedirs(save_path.format('new_obj'))
        except FileExistsError:
            pass

        test_new_obj_bulk(agent=rl_agent,
                              env=env,
                              config=config,
                              discount_factor=discount_factor,
                              save_path=save_path.format('new_obj/{}'))
        # test_new_obj_learning(agent=rl_agent,
        #                       env=env,
        #                       config=config,
        #                       discount_factor=discount_factor,
        #                       save_path=save_path)
    logging.info('Total time for new_obj loop : {}'.format(time.time() - toc))


def test(agent, env, config, test_number, discount_factor, save_path):

    # Setting the model into test mode (for dropout for example)
    agent.eval()
    env.eval()

    n_epochs_test = config["train_params"]["n_epochs_test"]

    try:
        os.makedirs(save_path.format('per_obj/'))
    except FileExistsError:
        pass

    lengths, rewards = [], []
    obj_type = config['env_type']['objective']['type']
    number_epochs_to_store = config['io']['num_epochs_to_store']

    if obj_type == 'fixed':
        train_objectives = [env.reward_position]
    else:
        # For now, test only on previously seen examples
        train_objectives = env.train_objectives

    for num_objective, objective in enumerate(train_objectives):
        logging.debug('Switching objective to {}'.format(objective))

        env.reward_position = objective

        for epoch in range(n_epochs_test):
            assert env.reward_position == objective, "position changed, warning"

            # WARNING FREEZE COUNT SO THE MAZE DOESN'T CHANGE
            env.count_ep_in_this_maze = 0
            env.count_current_objective = 0

            state = env.reset(show=False)

            done = False
            time_out = 20
            num_step = 0
            epoch_rewards = []
            video = []

            if epoch < number_epochs_to_store:
                video.append(env.render(show=False))

            while not done and num_step < time_out:
                num_step += 1
                action = agent.forward(state, 0.)
                next_state, reward, done, _info = env.step(action)

                assert bool(reward) == bool(done), "Holy shit."
                assert bool(reward) == bool(_info['agent_position'] == _info['reward_position']), "Epic Fail"

                if epoch < number_epochs_to_store:
                    video.append(env.render(show=False))

                epoch_rewards += [reward]
                state = next_state

            discount_factors = np.array([discount_factor**i for i in range(len(epoch_rewards))])
            rewards.append(np.sum(epoch_rewards*discount_factors))
            lengths.append(len(epoch_rewards))

            if epoch < number_epochs_to_store:
                if 'text' in obj_type:
                    make_video(video, save_path.format('test_{}_{}_{}'.format(test_number, num_objective, epoch)), state['objective'])
                else:
                    make_video(video, save_path.format('test_{}_{}_{}'.format(test_number, num_objective, epoch)), repr(_info))


        with open(save_path.format('per_obj/obj' + str(num_objective) + '_rewards.txt'), 'a+') as f:
            f.write("{}\n".format(np.mean(rewards)))

        with open(save_path.format('per_obj/obj' + str(num_objective) + '_lengths.txt'), 'a+') as f:
            f.write("{}\n".format(np.mean(lengths)))

    # Setting the model back into train mode (for dropout for example)
    agent.train()
    env.train()

    return np.mean(rewards), np.mean(lengths)


def test_zero_shot(agent, env, config, discount_factor, test_number, save_path):
    # Do it only once after fully training the model, otherwise will be
    # ridiculously slow

    # Setting the model into test mode (for dropout for example)
    agent.eval()
    env.eval()

    lengths, rewards = [], []
    obj_type = config['env_type']['objective']['type']
    number_epochs_to_store = config['io']['num_epochs_to_store']
    n_epochs_test = config["train_params"]["n_epochs_test"]

    if obj_type == 'fixed':
        # Do nothing for 'fixed' objective_type
        return
    elif 'image' in obj_type:
        # Here, test with untrained exit points
        test_objectives = env.test_objectives
    else:
        assert False, 'Objective {} not supported'.format(obj_type)

    for num_objective, objective in enumerate(test_objectives):
        logging.debug('Switching objective to {}'.format(objective))
        env.reward_position = objective
        for epoch in range(n_epochs_test):
            # WARNING FREEZE OBJECTIVE COUNT SO IT DOESN'T CHANGE
            env.count_current_objective = 0

            state = env.reset(show=False)

            done = False
            time_out = 20
            num_step = 0
            epoch_rewards = []
            video = []

            if epoch < number_epochs_to_store:
                video.append(env.render(show=False))

            while not done and num_step < time_out:
                num_step += 1
                action = agent.forward(state, 0.)
                next_state, reward, done, _ = env.step(action)

                if epoch < number_epochs_to_store:
                    video.append(env.render(show=False))

                epoch_rewards += [reward]
                state = next_state

            discount_factors = np.array([discount_factor**i for i in range(len(epoch_rewards))])
            rewards.append(np.sum(epoch_rewards*discount_factors))
            lengths.append(len(epoch_rewards))

            if epoch < number_epochs_to_store:
                make_video(video, save_path.format('test_zero_shot_{}_{}_{}'.format(test_number, num_objective, epoch)))

    # Setting the model back into train mode (for dropout for example)
    agent.train()
    env.train()

    return np.mean(rewards), np.mean(lengths)


def test_new_obj_learning(agent, env, config, discount_factor, save_path):
    env.train()

    lengths, rewards = [], []
    obj_type = config['env_type']['objective']['type']
    number_epochs_to_store = config['io']['num_epochs_to_store']

    n_epochs = config["train_params"]["n_epochs"]
    test_every = config["train_params"]["test_every"]

    logging.info(" ")
    logging.info("Begin new_obj_learning evaluation")
    logging.info("===============")

    if obj_type == 'fixed':
        # Do nothing for 'fixed' objective_type
        return
    else:
        test_objectives = env.test_objectives

    n_epochs_new_obj = n_epochs // len(env.train_objectives) * 2
    test_every_new_obj = test_every // len(env.train_objectives) * len(test_objectives) # To keep same total duration, not same number of points in each curve

    state_dict, saved_memory = agent.save_state()
    logging.info('Agent state saved')

    for num_objective, objective in enumerate(test_objectives):
        logging.debug('Switching objective to {}'.format(objective))
        env.reward_position = objective
        agent.load_state(state_dict, saved_memory)
        logging.info('Agent state loaded')

        eps_range = compute_epsilon_schedule(config['train_params'], n_epochs_new_obj)

        reward_list = []
        length_list = []

        for epoch in range(n_epochs_new_obj):
            env.count_current_objective = 0
            state = env.reset(show=False)

            done = False
            time_out = 20
            num_step = 0

            if epoch % test_every_new_obj == 0:
                env.base_folder = 'test/'
                rewards = []
                lengths = []
                for test_round in range(10):
                    env.count_current_objective = 0
                    state = env.reset(show=False)

                    done = False
                    time_out = 20
                    num_step = 0
                    epoch_rewards = []
                    video = []

                    while not done and num_step < time_out:
                        num_step += 1
                        action = agent.forward(state, 0.)
                        next_state, reward, done, _ = env.step(action)
                        state = next_state
                        epoch_rewards += [reward]
                        state = next_state

                    discount_factors = np.array([discount_factor**i for i in range(len(epoch_rewards))])
                    rewards.append(np.sum(epoch_rewards*discount_factors))
                    lengths.append(len(epoch_rewards))

                logging.info("Epoch {} new obj {} test : averaged reward {:.2f}, average length {:.2f}".format(epoch, num_objective, np.mean(rewards), np.mean(lengths)))
                reward_list.append(np.mean(rewards))
                length_list.append(np.mean(lengths))

                env.train()

            while not done and num_step < time_out:
                num_step += 1
                action = agent.forward(state, eps_range[epoch])
                next_state, reward, done, _ = env.step(action)
                loss = agent.optimize(state, action, next_state, reward)
                state = next_state

            agent.callback(epoch)

        try:
            os.makedirs(save_path.format('new_obj/'))
        except FileExistsError:
            pass
        save_stats(save_path.format('new_obj/' + str(num_objective) + '_{}'), reward_list, length_list)


def test_new_obj_bulk(agent, env, config, discount_factor, save_path):
    env.train()

    n_epochs = config["train_params"]["n_epochs"] // 3
    test_every = config["train_params"]["test_every"] // 3
    eps_range = compute_epsilon_schedule(config['train_params'], n_epochs)

    backup = env.train_objectives
    env.train_objectives = deepcopy(env.test_objectives)

    reward_list = []
    length_list = []


    for epoch in range(n_epochs):
        state = env.reset(show=False)
        done = False
        time_out = 20
        num_step = 0

        if epoch % test_every == 0:
            reward, length = test(agent=agent,
                                  env=env,
                                  config=config,
                                  test_number=epoch,
                                  discount_factor=discount_factor,
                                  save_path=save_path)

            logging.info("Epoch {} test : averaged reward {:.2f}, average length {:.2f}".format(epoch, reward, length))

            with open(save_path.format('train_lengths'), 'a+') as f:
                f.write("{} {}\n".format(epoch, length))
                length_list.append(length)
            with open(save_path.format('train_rewards'), 'a+') as f:
                f.write("{} {}\n".format(epoch, reward))
                reward_list.append(reward)
            make_eval_plot(save_path.format('train_lengths'), save_path.format('eval_curve.png'))
            make_eval_plot(save_path.format('train_rewards'), save_path.format('eval_curve_rew.png'))

        while not done and num_step < time_out:
            num_step += 1
            action = agent.forward(state, eps_range[epoch])
            next_state, reward, done, _info = env.step(action)

            assert bool(reward) == bool(done), "Holy shit."
            assert bool(reward) == bool(_info['agent_position'] == _info['reward_position']), "Epic Fail"

            loss = agent.optimize(state, action, next_state, reward)
            state = next_state

        agent.callback(epoch)
        env.post_process()

    env.train_objectives = backup
    save_stats(save_path, reward_list, length_list)




if __name__ == '__main__':

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-exp_dir", type=str, default="out", help="Directory with one expe")
    parser.add_argument("-env_config", type=str, help="Which file correspond to the experiment you want to launch ?")
    parser.add_argument("-model_config", type=str, help="Which file correspond to the experiment you want to launch ?")
    parser.add_argument("-env_extension", type=str, help="Do you want to override parameters in the env file ?")
    parser.add_argument("-model_extension", type=str, help="Do you want to override parameters in the model file ?")
    parser.add_argument("-display", type=str, help="Display images or not")
    parser.add_argument("-seed", type=int, default=0, help="Manually set seed when launching exp")
    parser.add_argument("-device", type=int, default=-1, help="Manually set GPU")

    args = parser.parse_args()

    # # Change your current wd so it's located in image_maze not src
    # current_wd_full = os.getcwd()
    # path, folder = os.path.split(current_wd_full)
    #
    # if folder != 'image_maze':
    #     os.chdir('../')

    full_train_test(env_config=args.env_config,
                    env_extension=args.env_extension,
                    model_config=args.model_config,
                    model_extension=args.model_extension,
                    device=args.device,
                    seed=args.seed,
                    exp_dir=args.exp_dir)
