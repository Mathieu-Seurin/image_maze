from keras.datasets import mnist, fashion_mnist
import numpy as np
from copy import copy

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


class ImageGridWorld(object):
    def __init__(self, config, show=False):

        self.n_row = config["n_row"]
        self.n_col = config["n_col"]
        self.position = []

        (self.digits_im, self.digits_labels), (_, _) = mnist.load_data()
        (self.fashion_im, self.fashion_labels), (_, _) = fashion_mnist.load_data()

        self.size_img = self.fashion_im[0].shape
        self.size_vector = self.size_img[0]*self.size_img[1]

        self.black_image = np.zeros((*self.size_img, 3))

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
            self.get_state = self.get_current_square
        elif config["state_type"] == "surrounding":
            self.get_state = self.get_current_square_and_all_directions
        else:
            #Todo all checkboard
            raise NotImplementedError("Need to implement all checkboard as state, but seems stupid")

        # ==================== Reward Type =======================
        # =========================================================
        # reward is only located at one place : fixed
        # TODO : have a reward that can change its location
        if config["reward_type"] == "fixed":
            self.get_reward = self.reward_fixed
        else:
            raise NotImplementedError("Need to change the reward location and add its position to the state")

    def reset(self):

        position_on_reward = True
        while position_on_reward:
            y = np.random.randint(self.n_row-1)
            x = np.random.randint(self.n_col-1)
            self.position = [x, y]
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

        self.position = [x,y]
        observation = self.get_state()
        reward = self.get_reward()

        if reward == 1:
            done = True
        else:
            done = False

        info = copy(self.position)

        return observation, reward, done, info

    def get_current_square_and_all_directions(self):

        x,y = self.position

        #A State correspond to : (current image, north, south, east, west) so 5 images => 15 channel
        # (may be too much)
        # TODO : Rescale image ?? Less channel ?
        self.observation = np.zeros((self.size_img[0], self.size_img[1], 15))

        self.observation[:, :, :3] = self.get_current_square()

        if x+1 < self.n_row : #North
            self.observation[:, :, 3:6] = self.grid[x+1, y]
        else:
            self.observation[:, :, 3:6] = self.black_image

        if x-1 >= 0 : #South
            self.observation[:, :, 6:9] = self.grid[x-1, y]
        else:
            self.observation[:, :, 6:9] = self.black_image

        if y-1 >= 0 : #East
            self.observation[:,:,9:12] = self.grid[x, y-1]
        else:
            self.observation[:, :,9:12] = self.black_image

        if y+1 < self.n_col : #West
            self.observation[:,:, 12:15] = self.grid[x, y+1]
        else:
            self.observation[:, :, 12:15] = self.black_image

        return self.observation

    def get_current_square(self):
        x,y = self.position
        return self.grid[x,y]

    def reward_fixed(self):
        if np.all(self.position == [self.n_row-1, self.n_col-1]):
            return 1
        else:
            return 0

    def render(self):

        custom_grid = np.copy(self.grid)
        if self.position != []:
            x,y = self.position
            x_size, y_size = self.size_img
            x_middle = x_size//2
            y_middle = y_size//2

            custom_grid[x,y, x_middle-3:x_middle+3, y_middle-3:y_middle+3, :] = [1,0,0]

        shown_grid = np.concatenate([custom_grid[i] for i in reversed(range(self.n_row))], axis=1)
        shown_grid = np.concatenate([shown_grid[i] for i in range(self.n_col)], axis=1)

        plt.figure()
        plt.imshow(shown_grid)
        plt.show()


    def create_grid_of_image(self, grid_type="all_diff", show=False):

        if grid_type == "all_diff":
            self.grid = np.zeros((self.n_row, self.n_col, self.size_img[0], self.size_img[1], 3))

            count = 0
            for i in range(self.n_row):
                for j in range(self.n_col):
                    background_color = self.background[i,j]
                    self.grid[i,j] = self.load_random_image_per_class(count, background_color, show=False)
                    count += 1
        else:
            raise NotImplementedError("Only all_diff is available at the moment")

        if show :
            self.render()

    def create_background(self, show=False):

        background = np.ones((self.n_row, self.n_col, 3))

        background[:, :, 0] = np.tile(np.linspace(0, 1, self.n_col), (self.n_row, 1))
        background[:, :, 2] = np.tile(np.linspace(1, 0, self.n_row), (self.n_col, 1)).T

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
        image_selected = dataset[random_image_id]

        #black_area = np.where(image_selected == 0)
        black_area = np.where(image_selected < 10)

        image_selected = self.to_rgb(image_selected)

        image_selected[black_area] = background_color

        if show:
            plt.figure()
            plt.imshow(image_selected)
            plt.show()

        return image_selected

    def to_rgb(self, im):
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.float32)
        ret[:, :, 0] = im/255
        ret[:, :, 1] = ret[:, :, 2] = ret[:, :, 0]
        return ret

    def action_space(self):
        return 4


if __name__ == "__main__":

    config = {"n_row":5, "n_col":4, "state_type":"surrounding", "reward_type":"fixed"}
    maze = ImageGridWorld(config=config, show=True)
    print(maze.size_vector)
    print(maze.size_img)