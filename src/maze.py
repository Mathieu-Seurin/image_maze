from keras.datasets import mnist, fashion_mnist
import numpy as np
from copy import copy
import random
import itertools

import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from image_utils import to_rgb_channel_first, to_rgb_channel_last, channel_last_to_channel_first, channel_first_to_channel_last, plot_single_image

class ImageGridWorld(object):
    def __init__(self, config, show=False):

        # Todo : load them from a file
        self.mean_per_channel = np.array([0.0441002, 0.0441002, 0.01040499])
        self.std_per_channel = np.array([0.19927699,  0.19927699,  0.08861534])


        self.n_row = config["n_row"]
        self.n_col = config["n_col"]
        self.position = []

        # Count the number of time the current objective
        self.count_current_objective = 0
        self.count_current_objective_success = 0

        (self.digits_im, self.digits_labels), (_, _) = mnist.load_data()
        (self.fashion_im, self.fashion_labels), (_, _) = fashion_mnist.load_data()

        self.size_img = self.fashion_im[0].shape
        self.size_vector = self.size_img[0]*self.size_img[1]

        self.black_image = np.zeros((3, *self.size_img))

        self.background = []
        self.create_background(show=False)

        self.grid = []
        self.create_grid_of_image(grid_type="all_diff", show=show)

        # ========== What does a state correspond to ? ===========
        # =========================================================
        # The square you're on ? ("current")
        # The square you're on + North South West East square ? ("surrounding")
        # All the checkboard ? ("all")
        if config["state_type"] == "current":
            self.get_env_state = self.get_current_square
        elif config["state_type"] == "surrounding":
            self.get_env_state = self.get_current_square_and_all_directions
        else:
            #Todo : can view only in front of him (change actions)
            raise NotImplementedError("Need to implement front view, maybe other maze")

        # ==================== Reward Type =======================
        # =========================================================
        # reward is only located at one place : fixed
        if config["objective"]["type"] == "fixed":
            self.reward_position = [2, 2]

            # Identity : Do nothing
            self.get_objective_state = lambda *args: None
            self.post_process = lambda *args: None

        else:
            objective_type = config["objective"]["type"]
            if  objective_type == "image":
                self.get_objective_state = self._get_image_objective

            elif objective_type == "text":
                self.get_objective_state = self._get_text_objective
            else:
                raise Exception(f"objective type must be 'fixed', 'text' or 'image', not {objective_type}")

            # Changing the number of state you're alternating with, and the speed at which you change objective
            #all_objective = list(itertools.product(range(self.n_row), range(self.n_col)))
            all_objective = [(1,0),(3,1),(4,3),(1,3)]
            random.shuffle(all_objective)
            self.n_objective = config["objective"]["curriculum"]["n_objective"]
            self.objective_changing_every = config["objective"]["curriculum"]["change_every"]

            self.objectives = all_objective[:self.n_objective]
            self.reward_position = random.sample(self.objectives, k=1)[0]

            self.post_process = self._change_objective


    def get_state(self):
        state = dict()
        state["env_state"] = self.get_env_state()
        state["objective"] = self.get_objective_state()
        return state

    def _get_image_objective(self):
        #Todo : Take images from the test_set as objective instead of the exact image
        x,y = self.reward_position
        return self.grid[x,y]

    def _get_text_objective(self):
        raise NotImplementedError("Not yet, image is only available at the moment")

    def _change_objective(self):
        if self.count_current_objective >= self.objective_changing_every:
            self.reward_position = random.sample(self.objectives, k=1)[0]
            self.count_current_objective = 0
        else:
            self.count_current_objective += 1

    def reset(self):

        position_on_reward = True
        while position_on_reward:
            y = np.random.randint(self.n_row-1)
            x = np.random.randint(self.n_col-1)
            self.position = (x, y)
            position_on_reward = bool(self.get_reward())
            # if position is on reward, change position

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

    def get_current_square_and_all_directions(self):

        x,y = self.position

        #A State correspond to : (current image, north, south, east, west) so 5 images => 15 channel
        # (may be too much)
        # TODO : Rescale image ?? Less channel ?
        self.observation = np.zeros((15, self.size_img[0], self.size_img[1]))

        self.observation[ :3, :, :] = self.get_current_square()

        if x+1 < self.n_row : #North
            self.observation[3:6, :, :] = self.grid[x+1, y]
        else:
            self.observation[3:6, :, :] = self.black_image

        if x-1 >= 0 : #South
            self.observation[6:9, :, :] = self.grid[x-1, y]
        else:
            self.observation[6:9, :, :] = self.black_image

        if y-1 >= 0 : #East
            self.observation[9:12, :, :] = self.grid[x, y-1]
        else:
            self.observation[9:12, :, :] = self.black_image

        if y+1 < self.n_col : #West
            self.observation[12:15, :, :] = self.grid[x, y+1]
        else:
            self.observation[12:15, :, :] = self.black_image

        return self.observation

    def get_current_square(self):
        x,y = self.position
        return self.grid[x,y]

    def get_reward(self):
        if np.all(self.position == self.reward_position):
            return 1
        else:
            return 0

    def render(self):
        """
        This function print the board and the position of the agent
        ONLY in this function, the image format is (H,W,C) (changed at the beginning)
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

        plt.figure()
        plt.imshow(shown_grid)
        plt.show()

    def create_grid_of_image(self, grid_type="all_diff", show=False):

        if grid_type == "all_diff":
            self.grid = np.zeros((self.n_row, self.n_col, 3, self.size_img[0], self.size_img[1]))
            grid_plot = np.zeros((self.n_row, self.n_col, self.size_img[0], self.size_img[1], 3))

            count = 0
            for i in range(self.n_row):
                for j in range(self.n_col):
                    background_color = self.background[:, i,j]
                    image_selected_channel_last = self.load_random_image_per_class(class_id=count,
                                                                      background_color=background_color,
                                                                      show=False)

                    grid_plot[i,j] = image_selected_channel_last

                    # Normalize image
                    image_selected_channel_last = self.normalize(image_selected_channel_last)
                    formatted_image = channel_last_to_channel_first(image_selected_channel_last)
                    self.grid[i, j] = formatted_image
                    count += 1

            self.grid_plot = grid_plot
            #self.create_grid_plot(grid_plot)

        else:
            raise NotImplementedError("Only all_diff is available at the moment")

        if show :
            self.render()

    def create_grid_plot(self, grid_plot):
        custom_grid = grid_plot

        # swap from x,y, c, h, w => x,y, h,w,c
        custom_grid = custom_grid.swapaxes(2,4)
        custom_grid = custom_grid.swapaxes(2,3)
        self.grid_plot = custom_grid

    def create_background(self, show=False):

        background = np.ones((3, self.n_row, self.n_col))

        background[0, :, :] = np.tile(np.linspace(0, 1, self.n_col), (self.n_row, 1))
        background[2, :, :] = np.tile(np.linspace(1, 0, self.n_row), (self.n_col, 1)).T

        self.background = background

        if show:
            plt.figure()
            plt.imshow(background)
            plt.show()

    def load_random_image_per_class(self, class_id, background_color, show=False):

        if class_id <= 9:
            labels = self.digits_labels
            dataset = self.digits_im
        else:
            labels = self.fashion_labels
            dataset = self.fashion_im
            class_id = class_id - 10

        random_image_id = np.random.choice(np.where(labels == class_id)[0])
        image_selected_grey = dataset[random_image_id]

        #black_area = np.where(image_selected == 0)
        black_area = np.where(image_selected_grey < 10)

        image_selected_channel_last = to_rgb_channel_last(image_selected_grey)
        image_selected_channel_last /= 255

        image_selected_channel_last[black_area] = background_color
        if show:
            plt.figure()
            plt.imshow(image_selected_channel_last)
            plt.show()

        return image_selected_channel_last


    def normalize(self, im):
        im = im - self.mean_per_channel[np.newaxis, np.newaxis, :]
        im = im / self.std_per_channel[np.newaxis, np.newaxis, :]
        return im

    def action_space(self):
        return 4


if __name__ == "__main__":

    config = {"n_row":5, "n_col":4, "state_type":"surrounding", "reward_type":"fixed"}
    maze = ImageGridWorld(config=config, show=True)
    print(maze.size_vector)
    print(maze.size_img)