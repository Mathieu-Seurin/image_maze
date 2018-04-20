from keras.datasets import mnist, fashion_mnist
import numpy as np
from copy import copy
import random
import itertools
import time
import matplotlib
import matplotlib.pyplot as plt
from image_utils import to_rgb_channel_first, to_rgb_channel_last, channel_last_to_channel_first, channel_first_to_channel_last, plot_single_image

# TODO : make test reproducible


class ImageGridWorld(object):
    def __init__(self, config, show=False):

        # Todo : load them from a file
        # See compute_norm_img.py for details
        self.mean_per_channel = np.array([0.5450519,   0.88200397,  0.54505189])
        self.std_per_channel = np.array([0.35243599, 0.23492979,  0.33889725])

        # self.mean_per_channel = np.array([0.5,   0.5,  0.5])
        # self.std_per_channel = np.array([0.5,   0.5,  0.5])

        self.n_row = config["n_row"]
        self.n_col = config["n_col"]
        self.position = []


        # The first tuple correspond to the "train" images that will be used for pretrain network
        # So we don't use them for the maze
        (_, _), (self.digits_im, self.digits_labels) = mnist.load_data()
        (_, _), (self.fashion_im, self.fashion_labels) = fashion_mnist.load_data()

        self.size_img = self.fashion_im[0].shape
        self.size_vector = self.size_img[0]*self.size_img[1]

        self.black_image = np.zeros((3, *self.size_img))

        self.background = []
        self.create_background(show=False)

        self.maze_grid = [] # Grid containing an image on each square of the board
        self.grid_class = [] # Grid containing the class of the state (0 : digit 0, 1: digit 1 ..., 19 : boots)
        self.current_objective = None

        self.grid_type = config["maze_type"]
        self.create_grid_of_image(show=show)

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
        if config["objective"]["modality"] == "fixed":
            self.reward_position = (2, 2)

            # Identity : Do nothing
            self.get_objective_state = lambda *args: None
            self.post_process = lambda *args: None

        else:
            objective_type = config["objective"]["modality"]
            if  objective_type == "image":
                self.get_objective_state = self._get_image_objective

                # Can be "case" or "category"
                # "case" means : the objective_state is the image of a checkboard case ("GO FOR THIS T-SHIRT")
                # "category" means : the objective_state is not the EXACT state you need to reach, but sampled from the same class
                # ("GO FOR A T-SHIRT (even though it's not the same as on the picture)")
                self.objective_image_type = config["objective"]["objective_image"]


            elif objective_type == "text":
                self.get_objective_state = self._get_text_objective
            else:
                raise Exception("objective type must be 'fixed', 'text' or 'image', not {}".format(objective_type))

            # Changing the number of state you're alternating with, and the speed at which you change objective
            #all_objective = list(itertools.product(range(self.n_row), range(self.n_col)))
            all_objective = [(1,0),(3,1),(4,3),(1,3)]
            random.shuffle(all_objective)
            self.n_objective = config["objective"]["curriculum"]["n_objective"]
            self.objective_changing_every = config["objective"]["curriculum"]["change_every"]

            self.objectives = all_objective[:self.n_objective]
            self.reward_position = random.sample(self.objectives, k=1)[0]

            self.post_process = self._change_objective

        # ==================== Maze Type ==========================
        # =========================================================

        # Count the number of time the current objective
        self.count_current_objective = 0  # To enable changing objectives every 'n' step
        self.count_ep_in_this_maze = 0  # To change maze every 'n' step
        self.change_maze_every_n = float("inf") if config["change_maze"]==0 else config["change_maze"]

    def get_state(self):
        state = dict()
        state["env_state"] = self.get_env_state()
        state["objective"] = self.get_objective_state()
        return state

    def _get_image_objective(self):

        x,y = self.reward_position
        if self.current_objective is None:

            if self.objective_image_type == "case":
                img_objective = self.maze_grid[x, y]

            elif self.objective_image_type == "category" :
                objective_class = self.grid_class[x,y]
                background_color = self.background[:, x,y]
                img_objective, _ = self.load_random_image_per_class(class_id=objective_class,
                                                                 background_color=background_color,
                                                                 show=True)
            else:
                raise NotImplementedError("Wrong type of objective image type")
            self.current_objective = img_objective
        return self.current_objective


    def _get_text_objective(self):
        raise NotImplementedError("Not yet, image is only available at the moment")

    def _change_objective(self):
        if self.count_current_objective >= self.objective_changing_every:
            self.reward_position = random.sample(self.objectives, k=1)[0]
            self.count_current_objective = 0
            self.current_objective = None
        else:
            self.count_current_objective += 1

    def reset(self, show=True):
        # Change maze every n step
        if self.count_ep_in_this_maze >= self.change_maze_every_n:
            self.create_grid_of_image(show=show)
            self.count_ep_in_this_maze = 0
        else:
            self.count_ep_in_this_maze += 1

        # Set begin position at random, not on the goal
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
            self.observation[3:6, :, :] = self.maze_grid[x + 1, y]
        else:
            self.observation[3:6, :, :] = self.black_image

        if x-1 >= 0 : #South
            self.observation[6:9, :, :] = self.maze_grid[x - 1, y]
        else:
            self.observation[6:9, :, :] = self.black_image

        if y-1 >= 0 : #East
            self.observation[9:12, :, :] = self.maze_grid[x, y - 1]
        else:
            self.observation[9:12, :, :] = self.black_image

        if y+1 < self.n_col : #West
            self.observation[12:15, :, :] = self.maze_grid[x, y + 1]
        else:
            self.observation[12:15, :, :] = self.black_image

        return self.observation

    def get_current_square(self):
        x,y = self.position
        return self.maze_grid[x, y]

    def get_reward(self):
        if np.all(self.position == self.reward_position):
            return 1
        else:
            return 0

    def render(self, display=False):
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

        if display:
            plt.figure()
            plt.imshow(shown_grid)
            plt.show()
            time.sleep(1)

        shown_grid = shown_grid.transpose([2, 0, 1])
        return shown_grid

    def create_grid_of_image(self, show=False):

        grid_type = self.grid_type
        if grid_type == "sequential":
            self.maze_grid = np.zeros((self.n_row, self.n_col, 3, self.size_img[0], self.size_img[1]))
            self.grid_class = np.zeros((self.n_row, self.n_col))
            self.grid_plot = np.zeros((self.n_row, self.n_col, self.size_img[0], self.size_img[1], 3))

            count = 0
            for i in range(self.n_row):
                for j in range(self.n_col):
                    background_color = self.background[:, i,j]
                    image_normalized_channel_first, image_display_channel_last = self.load_random_image_per_class(class_id=count,
                                                                      background_color=background_color,
                                                                      show=False)
                    # save image in format (h,w,c)
                    self.grid_plot[i,j] = image_display_channel_last
                    self.maze_grid[i, j] = image_normalized_channel_first

                    self.grid_class[i, j] = count # to indicate what is the class of the image present in this case
                    count += 1

        else:
            # Todo : TSNE order
            raise NotImplementedError("Only sequential grid is available at the moment")

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

        image_display_channel_last = to_rgb_channel_last(image_selected_grey)
        image_display_channel_last /= 255

        image_display_channel_last[black_area] = background_color
        if show:
            plt.figure()
            plt.imshow(image_display_channel_last)
            plt.show()

        img_normalized = self.normalize(np.copy(image_display_channel_last))
        img_normalized_channel_first = channel_last_to_channel_first(img_normalized)

        return img_normalized_channel_first, image_display_channel_last


    def normalize(self, im):

        assert im.shape[2] == 3, "rbg channel need to be last"

        # plot_single_image(im)

        im[:,:,0] = (im[:,:,0] - self.mean_per_channel[0]) / self.std_per_channel[0]
        im[:,:,1] = (im[:,:,1] - self.mean_per_channel[1]) / self.std_per_channel[1]
        im[:,:,2] = (im[:,:,2] - self.mean_per_channel[2]) / self.std_per_channel[2]

        # plot_single_image(im)

        return im

    def action_space(self):
        return 4


if __name__ == "__main__":

    from torchvision import transforms

    config = {"n_row":5,
              "n_col":4,
              "state_type":"surrounding",
              "maze_type":"sequential",
              "change_maze": 0,
              "objective":{
                  "modality":"fixed"
                }
              }
    maze = ImageGridWorld(config=config, show=True)

    im = np.random.random((2,2,3))
    save = np.copy(im)
    print("im =", im)

    ret = maze.normalize(im)
    print("perso = ",ret)

    im = np.copy(save)

    pipe = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(maze.mean_per_channel, maze.std_per_channel)
        ])

    ret = pipe(img=im*255)
    print("pytorch = ", ret)

    config = {"n_row": 5,
              "n_col": 4,
              "state_type": "surrounding",
              "maze_type": "sequential",
              "change_maze": 0,
              "objective": {
                  "curriculum":{
                      "n_objective" : 2,
                      "change_every" : 2
                  },
                  "modality": "image",
                  "objective_image" : "category"

                }
              }
    maze = ImageGridWorld(config=config, show=True)


    maze._get_image_objective()
    maze.position = (0,0)
    maze.render(display=True)