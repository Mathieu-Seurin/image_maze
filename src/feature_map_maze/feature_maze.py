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

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

from PIL import Image

# For debugging, not actually necessary
def plot_single_image(im):
    if im.shape[0] == 3 :
        im = channel_first_to_channel_last(im)
    plt.figure()
    plt.imshow(im)
    plt.show()



class ImageFmapGridWorld(object):
    def __init__(self, config):
        # Use image normalization after loading?
        self.mean_per_channel = np.array([0.5450519,   0.88200397,  0.54505189])
        self.std_per_channel = np.array([0.35243599, 0.23492979,  0.33889725])
        self.use_normalization = (config['use_normalization'] == 'True')

        self.n_row = config["n_row"]
        self.n_col = config["n_col"]
        self.size_img = (28, 28)
        self.position = []

        self.grid = []
        self.preprocessed_grid = []
        self.grid_type = config["maze_type"]
        self.create_grid_of_image(show=False)
        self.preproc_state = (config['preproc_state'] == 'True')

        if config["state_type"] == "current":
            self.get_env_state = lambda: self.get_current_square(preproc = self.preproc_state)
        elif config["state_type"] == "surrounding":
            self.get_env_state = lambda: self.get_current_square_and_all_directions(preproc = self.preproc_state)
        else:
            #Todo : can view only in front of him (change actions)
            raise NotImplementedError("Need to implement front view, maybe other maze")


        if config["objective"]["type"] == "fixed":
            self.get_objective_state = lambda *args: None
            self.reward_position = (2, 2)
            self.post_process = lambda *args: None
        else:
            objective_type = config["objective"]["type"]
            if  objective_type == "same_image":
                # Exactly the same image as on the exit
                self.get_objective_state = self._get_same_image_objective
            elif  objective_type == "random_image":
                # Random image with color from same class as exit
                self.get_objective_state = self._get_random_image_objective
            elif  objective_type == "random_image_no_bkg":
                # Random image from same class as exit without background
                self.get_objective_state = self._get_random_image_no_bkg_objective
            elif  objective_type == "preprocessed_image":
                # Random fmap from same class as exit with background
                self.get_objective_state = self._get_preprocessed_image_objective
            elif  objective_type == "preprocessed_image_no_bkg":
                # Random fmap from same class as exit without background
                self.get_objective_state = self._get_preprocessed_image_no_bkg_objective
            elif objective_type == "text":
                # Text description (should add this to the datasets)
                self.get_objective_state = self._get_text_objective
            else:
                raise Exception("Objective type {} not defined".format(objective_type))

            # Number of objectives / frequency of change
            self.n_objectives = config["objective"]["curriculum"]["n_objective"]
            self.objective_changing_every = config["objective"]["curriculum"]["change_every"]

            # Construct the list of objectives (the order is always the same)
            self.all_objectives = list(itertools.product(range(self.n_row), range(self.n_col)))
            objective_shuffler = random.Random(777)
            objective_shuffler.shuffle(self.all_objectives)
            self.objectives = self.all_objectives[:self.n_objectives]

            self.reward_position = (2, 2)
            self.post_process = self._change_objective


        self.count_current_objective = 0  # To enable changing objectives every 'n' step
        self.count_ep_in_this_maze = 0  # To change maze every 'n' step
        self.change_maze_every_n = float("inf") if config["change_maze"]==0 else config["change_maze"]


    def get_state(self):
        state = dict()
        state["env_state"] = self.get_env_state()
        state["objective"] = self.get_objective_state()
        return state

    def load_image_or_fmap(self, folder=None, class_id=None, seed=None, preproc=False):
        # With preproc=True, load a feature_map not an image
        # Returned object is FloatTensor
        path_to_images = folder.format(class_id)
        loader_local_rand = np.random.RandomState(seed=seed)
        all_files = os.listdir(path_to_images)

        ext = 'jpg' if not preproc else 'tch'
        filtered_files = [fn for fn in all_files if fn.split('.')[-1] == ext]

        chosen_file = loader_local_rand.choice(filtered_files)
        if preproc:
            return torch.load(path_to_images + '/{}'.format(chosen_file))
        else:
            return ToTensor()(Image.open(path_to_images + '/{}'.format(chosen_file)))


    def _get_same_image_objective(self):
        x,y = self.reward_position
        return FloatTensor(self.grid[x,y])

    def _get_random_image_objective(self):
        x,y = self.reward_position
        cat = x * self.n_col + y
        tmp = np.random.randint(2**16)
        img = self.load_image_or_fmap(folder='obj_images/with_bkg/{}', class_id=cat, seed=tmp)
        # plot_single_image(img)
        return img

    def _get_random_image_no_bkg_objective(self):
        x,y = self.reward_position
        cat = x * self.n_col + y
        tmp = np.random.randint(2**16)
        img = self.load_image_or_fmap(folder='obj_images/without_bkg/{}', class_id=cat, seed=tmp)
        # plot_single_image(img)
        return img

    def _get_preprocessed_image_objective(self):
        x,y = self.reward_position
        cat = x * self.n_col + y
        tmp = np.random.randint(2**16)
        fmap = self.load_image_or_fmap(folder='obj_images/with_bkg/{}', preproc=True, class_id=cat, seed=tmp)
        # print(fmap.shape)
        return fmap

    def _get_preprocessed_image_no_bkg_objective(self):
        x,y = self.reward_position
        cat = x * self.n_col + y
        tmp = np.random.randint(2**16)
        fmap = self.load_image_or_fmap(folder='obj_images/without_bkg/{}', preproc=True, class_id=cat, seed=tmp)
        # print(fmap.shape)
        return fmap

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
        if action == 0: # NORTH
            x = min(self.n_row-1, x+1)
        elif action == 1:  # SOUTH
            x = max(0, x-1)
        elif action == 2:  # EAST
            y = max(0, y-1)
        elif action == 3:  # WEST
            y = min(self.n_col-1, y+1)
        else:
            assert False, "Wrong action"

        self.position = (x,y)
        observation = self.get_state()
        reward = self.get_reward()

        if reward == 1:
            done = True
            self.post_process()
        else:
            done = False

        info = copy(self.position)
        return observation, reward, done, info

    def get_current_square(self, preproc=False):
        x,y = self.position
        if preproc:
            return FloatTensor(self.preprocessed_grid[x,y])
        else:
            return FloatTensor(self.grid[x,y])

    def get_square(self, x, y, preproc=False):
        x,y = self.position
        if preproc:
            return FloatTensor(self.preprocessed_grid[x,y])
        else:
            return FloatTensor(self.grid[x,y])

    def get_current_square_and_all_directions(self, preproc=False):
        x,y = self.position
        all_directions_obs = [self.get_current_square(preproc=preproc),]

        if x+1 < self.n_row : # North
            all_directions_obs.append(self.get_square(x+1, y, preproc=preproc))
        else:
            all_directions_obs.append(torch.zeros(all_directions_obs[0].shape))

        if x-1 >= 0 : # South
            all_directions_obs.append(self.get_square(x-1, y, preproc=preproc))
        else:
            all_directions_obs.append(torch.zeros(all_directions_obs[0].shape))

        if y-1 >= 0 : # East
            all_directions_obs.append(self.get_square(x, y-1, preproc=preproc))
        else:
            all_directions_obs.append(torch.zeros(all_directions_obs[0].shape))

        if y+1 < self.n_col : # West
            all_directions_obs.append(self.get_square(x, y+1, preproc=preproc))
        else:
            all_directions_obs.append(torch.zeros(all_directions_obs[0].shape))

        # Ensure everyone is on GPU
        all_directions_obs = tuple([x.cuda() for x in all_directions_obs])

        return torch.cat(all_directions_obs)


    def get_reward(self):
        if np.all(self.position == self.reward_position):
            return 1
        else:
            return 0

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
            plt.imshow(shown_grid)
            plt.show()
            time.sleep(1)

        shown_grid = shown_grid.transpose([2, 0, 1])
        return shown_grid

    def create_grid_of_image(self, show=False):
        grid_type = self.grid_type
        if grid_type == "sequential":
            self.grid = np.zeros((self.n_row, self.n_col, 3, 28, 28))
            # TODO : make this less hard-wired...
            self.grid_preprocessed = np.zeros((self.n_row, self.n_col, 32, 7, 7))
            self.grid_plot = np.zeros((self.n_row, self.n_col, 28, 28, 3))

            count = 0
            for i in range(self.n_row):
                for j in range(self.n_col):
                    plop = np.random.randint(2**16)
                    raw_img = self.load_image_or_fmap(folder='maze_images/{}' ,class_id=count, seed=plop)
                    fmap = self.load_image_or_fmap(folder='maze_images/{}', class_id=count, seed=plop, preproc=True)
                    self.grid_preprocessed[i,j] = fmap

                    # Images retrieved as tensors in (0,1)
                    self.grid_plot[i,j] = raw_img.cpu().numpy().transpose((1, 2, 0))
                    # Won't do anything unless specified in config
                    img = self.normalize(raw_img)
                    self.grid[i,j] = img
                    count += 1
        else:
            # Todo : TSNE order
            raise NotImplementedError("Only all_diff is available at the moment")
        if show :
            self.render()

    def action_space(self):
        return 4

    def normalize(self, im):
        if not self.use_normalization:
            return im
        im[:,:,0] = (im[:,:,0] - self.mean_per_channel[0]) / self.std_per_channel[0]
        im[:,:,1] = (im[:,:,1] - self.mean_per_channel[1]) / self.std_per_channel[1]
        im[:,:,2] = (im[:,:,2] - self.mean_per_channel[2]) / self.std_per_channel[2]
        return im

if __name__ == "__main__":
    config = {"n_row":5,
              "n_col":4,
              "state_type": "surrounding",
              "change_maze": 0,
              "preproc_state": "False",
              "use_normalization": "False",
              "maze_type": 'sequential',
              "objective":{
                  "type":"random_image_no_bkg",
                  "curriculum":{
                  "n_objective":3,
                  "change_every":10
                  }
                }
              }
    maze = ImageFmapGridWorld(config=config)
    for i in range(20):
        maze.reset()
        maze.render(show=True)
