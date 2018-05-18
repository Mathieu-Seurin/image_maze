from keras.datasets import mnist, fashion_mnist
import numpy as np
from copy import copy
import random
import itertools
import time
import matplotlib
import matplotlib.pyplot as plt
import os
from torchvision.transforms import ToTensor, ToPILImage
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from image_utils import channel_first_to_channel_last, channel_last_to_channel_first
from rl_agent.gpu_utils import use_cuda, FloatTensor, LongTensor, ByteTensor, Tensor
from PIL import Image

# For debugging, not actually necessary
def plot_single_image(im):
    if im.shape[0] == 3 :
        im = channel_first_to_channel_last(im)
    plt.figure()
    plt.imshow(im)
    plt.show()

if not os.path.isdir('./maze_images/0'):
    assert False, 'Please run preprocess in the src folder to generate datasets'

class ImageFmapGridWorld(object):
    def __init__(self, config, pretrained_features, save_image):
        # Use image normalization after loading?
        self.mean_per_channel = np.array([0.5450519,   0.88200397,  0.54505189])
        self.std_per_channel = np.array([0.35243599, 0.23492979,  0.33889725])

        # if you want to save replay or not, to save computation
        self.save_image = save_image

        # Just to remember the last images you loaded
        self.last_chosen_file = ''

        # If you want to use pretrain_features or raw images as state/objective
        self.preproc_state = pretrained_features

        self.n_objectives = 1 # Default
        self.is_multi_objective = False #Default
        self.use_normalization = config['use_normalization']

        self.n_row = config["n_row"]
        self.n_col = config["n_col"]
        self.size_img = (28, 28)
        self.position = []

        self.grid = []
        self.grid_type = config["maze_type"]
        self.create_grid_of_image(show=False)

        if config["state_type"] == "current":
            self.get_env_state = self.get_current_square
        elif config["state_type"] == "surrounding":
            self.get_env_state = self.get_current_square_and_all_directions
        else:
            #Todo : can view only in front of him (change actions)
            raise NotImplementedError("Need to implement front view, maybe other maze")


        if config["objective"]["type"] == "fixed":
            self.get_objective_state = lambda *args: None
            self.reward_position = (2, 2)
            self.post_process = lambda *args: None
        else:
            objective_type = config["objective"]["type"]

            self.get_objective_state = self.load_image_or_fmap

            if objective_type == "same_image":
                # Exactly the same image as on the exit
                self.get_objective_state = self.get_current_objective
            elif objective_type == "random_image":
                # Random image with color from same class as exit
                self.image_folder = 'obj_images/with_bkg/{}'
            elif objective_type == "random_image_no_bkg":
                self.image_folder = 'obj_images/without_bkg/{}'
            elif objective_type == "text":
                # Text description (should add this to the datasets)
                self.get_objective_state = self._get_text_objective
            else:
                raise Exception("Objective type {} not defined".format(objective_type))

            # Number of objectives / frequency of change
            self.n_objectives = config["objective"]["curriculum"]["n_objective"]
            self.is_multi_objective = self.n_objectives > 1
            self.objective_changing_every = config["objective"]["curriculum"]["change_every"]

            # Construct the list of objectives (the order is always the same)
            self.all_objectives = list(itertools.product(range(self.n_row), range(self.n_col)))
            objective_shuffler = random.Random(777)
            objective_shuffler.shuffle(self.all_objectives)
            self.objectives = self.all_objectives[:self.n_objectives]

            self.reward_position = (4, 3)
            self.post_process = self._change_objective


        self.count_current_objective = 0  # To enable changing objectives every 'n' step
        self.count_ep_in_this_maze = 0  # To change maze every 'n' step
        self.change_maze_every_n = float("inf") if config["change_maze"]==0 else config["change_maze"]

        if config["time_penalty"]:
            self.get_reward = self._get_reward_with_penalty
        else:
            self.get_reward = self._get_reward_no_penalty

    def get_state(self):
        state = dict()
        state["env_state"] = self.get_env_state()
        state["objective"] = self.get_objective_state()
        return state

    def get_current_objective(self):
        x,y = self.reward_position
        return Tensor(self.grid[x,y])

    def load_image_or_fmap(self, class_id=None, folder=None, preproc=None, raw=None, use_last_chosen_file=None):
        # Default is given by type of env, but can be overridden
        if preproc is None: preproc = self.preproc_state
        if folder is None: folder = self.image_folder

        if class_id is None:
            x, y = self.reward_position
            class_id = y + x * self.n_col

        ext = 'jpg' if not preproc else 'tch'

        path_to_images = folder.format(class_id)
        all_files = os.listdir(path_to_images)
        filtered_files = [fn for fn in all_files if fn.split('.')[-1] == ext]

        # That way, you can reuse the function for both the raw and preproc image
        if use_last_chosen_file:
            chosen_file = self.last_chosen_file+'.'+ext
        else:
            chosen_file = np.random.choice(filtered_files)
            #Keep only the id
            self.last_chosen_file = chosen_file.split('.')[0]

        # Not preproc
        if raw or not preproc:
            img = Image.open(path_to_images + '/{}'.format(chosen_file))
            if not raw:
                # Images retrieved as tensors in (0,1)
                img = channel_last_to_channel_first(np.array(img))
                # Won't do anything unless specified in config
                img = FloatTensor(self.normalize(img))

        # preproc is True : Use pretrain images
        else:
            img = torch.load(path_to_images + '/{}'.format(chosen_file)).type(Tensor)

        return img

    def _get_text_objective(self):
        raise NotImplementedError("Not yet, image is only available at the moment")

    def _change_objective(self):
        if self.count_current_objective >= self.objective_changing_every:
            self.reward_position = self.objectives[np.random.randint(len(self.objectives))]
            self.count_current_objective = 1
        else:
            self.count_current_objective += 1

    def reset(self, show=False):
        if self.count_ep_in_this_maze >= self.change_maze_every_n:
            self.create_grid_of_image(show=show)
            self.count_ep_in_this_maze = 1
        else:
            self.count_ep_in_this_maze += 1

        position_on_reward = True
        while position_on_reward:
            y = np.random.randint(self.n_col)
            x = np.random.randint(self.n_row)
            self.position = (x, y)
            position_on_reward = bool(self.get_reward())
        return self.get_state()

    def step(self, action):
        x,y = self.position
        reward = 0

        # Discourage invalid moves?
        wrong_action_penalty = 0.0

        if action == 0: # NORTH
            x_ = min(self.n_row-1, x+1)
            if x_ == x:
                reward -= wrong_action_penalty
            else:
                x = x_
        elif action == 1:  # SOUTH
            x_ = max(0, x-1)
            if x_ == x:
                reward -= wrong_action_penalty
            else:
                x = x_
        elif action == 2:  # EAST
            y_ = max(0, y-1)
            if y_ == y:
                reward -= wrong_action_penalty
            else:
                y = y_
        elif action == 3:  # WEST
            y_ = min(self.n_col-1, y+1)
            if y_ == y:
                reward -= wrong_action_penalty
            else:
                y = y_
        else:
            assert False, "Wrong action"

        self.position = (x,y)
        observation = self.get_state()
        reward += self.get_reward()

        if reward == 1:
            done = True
            self.post_process()
        else:
            done = False

        info = copy(self.position)
        return observation, reward, done, info

    def get_current_square(self):
        x,y = self.position
        return self.grid[x,y]

    def get_square(self, x, y):
        x,y = self.position
        return self.grid[x,y]

    def get_current_square_and_all_directions(self):
        x,y = self.position
        all_directions_obs = [self.get_current_square()]

        if x+1 < self.n_row : # North
            all_directions_obs.append(self.get_square(x+1, y))
        else:
            all_directions_obs.append(np.zeros(all_directions_obs[0].shape))

        if x-1 >= 0 : # South
            all_directions_obs.append(self.get_square(x-1, y))
        else:
            all_directions_obs.append(np.zeros(all_directions_obs[0].shape))

        if y-1 >= 0 : # East
            all_directions_obs.append(self.get_square(x, y-1))
        else:
            all_directions_obs.append(np.zeros(all_directions_obs[0].shape))

        if y+1 < self.n_col : # West
            all_directions_obs.append(self.get_square(x, y+1))
        else:
            all_directions_obs.append(np.zeros(all_directions_obs[0].shape))

        all_directions_obs = [Tensor(obs) for obs in all_directions_obs]

        return torch.cat(all_directions_obs, 0)

    def _get_reward_no_penalty(self):
        if np.all(self.position == self.reward_position):
            return 1
        else:
            return 0

    def _get_reward_with_penalty(self):
        if np.all(self.position == self.reward_position):
            return 1
        else:
            return -0.1

    def render(self, show=False):
        """
        This function print the board and the position of the agent
        ONLY in this function, the image format is (H,W,C)
        """

        custom_grid = np.copy(self.grid_plot)

        if self.position != []:
            x,y = self.position
            x_size, y_size = self.size_img
            x_middle = x_size//2
            y_middle = y_size//2

            x_rew, y_rew = self.reward_position

            # Display agent position as a red point.
            custom_grid[x,y, x_middle-3:x_middle+3, y_middle-3:y_middle+3, :] = [1,0,0]

            # Display reward position as a green point.
            custom_grid[x_rew,y_rew, x_middle-3:x_middle+3, y_middle-3:y_middle+3, :] = [0,1,0]

        shown_grid = np.concatenate([custom_grid[i] for i in reversed(range(self.n_row))], axis=1)
        shown_grid = np.concatenate([shown_grid[i] for i in range(self.n_col)], axis=1)

        if show:
            plt.figure()
            plt.imshow(shown_grid.astype(int))
            plt.show()
            time.sleep(1)

        shown_grid = shown_grid.transpose([2, 0, 1])
        return shown_grid

    def create_grid_of_image(self, show=False):
        grid_type = self.grid_type

        if grid_type == "sequential":

            # TODO : make this less hard-wired...
            if self.preproc_state:
                self.grid = np.zeros((self.n_row, self.n_col, 32, 7, 7))
            else:
                self.grid = np.zeros((self.n_row, self.n_col, 3, 28, 28))

            self.grid_plot = np.zeros((self.n_row, self.n_col, 28, 28, 3))

            count = 0
            for i in range(self.n_row):
                for j in range(self.n_col):
                    raw_img = self.load_image_or_fmap(folder='./maze_images/{}', class_id=count, preproc=False, raw=True)
                    if self.save_image:
                        self.grid_plot[i,j] = raw_img
                        use_last_chosen_file = True
                    else:
                        use_last_chosen_file = False
                    self.grid[i,j] = self.load_image_or_fmap(folder='./maze_images/{}', class_id=count, use_last_chosen_file=use_last_chosen_file)

                    count += 1
        else:
            # Todo : TSNE order
            raise NotImplementedError("Only all_diff is available at the moment")
        if show :
            self.render()

    def action_space(self):
        return 4

    def normalize(self, im):
        if self.use_normalization:
            im[:,:,0] = (im[:,:,0] - self.mean_per_channel[0]) / self.std_per_channel[0]
            im[:,:,1] = (im[:,:,1] - self.mean_per_channel[1]) / self.std_per_channel[1]
            im[:,:,2] = (im[:,:,2] - self.mean_per_channel[2]) / self.std_per_channel[2]
        return im

    def state_objective_dim(self):
        self.reset(show=False)
        state = self.get_state()
        state_objective_dim_dict = dict()
        state_objective_dim_dict['env_state'] = state['env_state'].shape
        state_objective_dim_dict['objective'] = state['objective'].shape
        concatened_dim = state_objective_dim_dict['env_state'][0] + state_objective_dim_dict['objective'][0]
        state_objective_dim_dict['concatenated'] = (concatened_dim, state_objective_dim_dict['env_state'][1], state_objective_dim_dict['env_state'][2])

        return state_objective_dim_dict

if __name__ == "__main__":
    config = {"n_row":5,
              "n_col":4,
              "state_type": "surrounding",
              "change_maze": 3,
              "preproc_state": "False",
              "use_normalization": "False",
              "maze_type": 'sequential',
              "objective":{
                  "type":"random_image_no_bkg",
                  "curriculum":{
                  "n_objective":3,
                  "change_every":2
                  }
                }
              }
    maze = ImageFmapGridWorld(config=config)
    for i in range(20):
        maze.reset()
        maze.render(show=True)
