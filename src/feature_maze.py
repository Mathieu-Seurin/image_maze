import numpy as np
from copy import copy
import random
import itertools
import time
import matplotlib
import matplotlib.pyplot as plt
import os

from image_text_utils import channel_first_to_channel_last, channel_last_to_channel_first, load_image_or_fmap, np_color_to_str, TextObjectiveGenerator

# For debugging, not actually necessary
def plot_single_image(im):
    if im.shape[0] == 3 :
        im = channel_first_to_channel_last(im)
    plt.figure()
    plt.imshow(im)
    plt.show()

class ImageFmapGridWorld(object):

    @property
    def path_to_maze_images(self):
        #                                                                     label    color   features type
        return self.image_location + self.train_test_folder + 'maze_images/' + '{}/' + '{}/' + '{}/'

    @property
    def path_to_objectives_images(self):
        #                                                                     class    color
        return self.image_location + self.train_test_folder + 'obj_images/'   + '{}/' + '{}/' + self.features_type

    @property
    def reward_position(self):
        return self._reward_position
    @reward_position.setter
    def reward_position(self, x):
        self.current_objective = None
        self._reward_position = x

    def __init__(self, config, features_type, save_image):

        # If in train mode, use images from train folder
        # Has to be changed from main when going to test mode

        # if you want to save replay or not, to save computation
        self.save_image = save_image

        self.n_objectives = 1 # Default
        self.is_multi_objective = False #Default

        # By default, images features are located in src/
        self.image_location = 'src/'
        self.train_test_folder = 'train/'

        # Can be 'normalized' (no pretrain), 'specific' or 'image_net'
        self.features_type = features_type
        if self.features_type == 'normalized':
            self.features_shape = (3, 28, 28)
        elif self.features_type == 'specific':
            self.features_shape = (32, 7, 7)
        elif self.features_type == 'image_net':
            raise NotImplementedError("No image net now")
        else:
            assert False, "Bad features type : {} Expecting 'normalized' (no pretrain), 'specific' or 'image_net'".format(self.features_type)

        # Do you use background or not ?
        #================================
        self.grid_type = config["maze_type"]
        self.use_background_for_state = 'no_bkg' not in self.grid_type
        self.objective_type = config["objective"]["type"]
        self.use_background_for_objective = 'no_bkg' not in self.objective_type

        # For text based objective, creating different "zone" aka colored room where the object are located
        if 'zone' in self.grid_type:
            self.n_zone = config['n_zone']
            assert self.n_zone%4 == 0, "The number of zone need to be a multiple of 4, so we have square maze"


        if not os.path.isdir(os.path.join(self.image_location, self.train_test_folder)):
            print(os.path.join(self.image_location, self.train_test_folder))
            assert False, 'Please run preprocess in the src folder to generate datasets'

        self.n_row = config["n_row"]
        self.n_col = config["n_col"]

        self.agent_position = []

        # grid containing the images
        self.grid = []
        # grid containing the class and color of each case
        self.grid_label_color = []
        self.current_objective = None
        self.create_grid_of_image(show=False)

        if config["state_type"] == "current":
            self.get_env_state = self.get_current_square
        elif config["state_type"] == "surrounding":
            self.get_env_state = self.get_current_square_and_all_directions
        else:
            #Todo : can view only in front of him (change actions)
            raise NotImplementedError("Need to implement front view, maybe other maze")


        #============== OBJECTIVE SPECIFICATION ==============
        #=====================================================
        if self.objective_type == 'fixed' :
            self.get_objective_state = lambda *args: None
            self._reward_position = (2, 2)
            self.post_process = lambda *args: None
        else:

            if self.objective_type == "same_image":
                # Exactly the same image as on the exit
                self.get_objective = self.get_current_objective

            elif 'text' in self.objective_type:

                if config["objective"]['text_difficulty'] == "easy":
                    sentence_file = "sentences_template_easy.txt"
                    print("Loading easy sentences")
                elif config["objective"]['text_difficulty'] == "hard":
                    sentence_file = "sentences_template_full.txt"
                    print("Loading hard sentences")
                else:
                    raise NotImplementedError("{} mode doesn't exist".format(config['objective']['text_difficulty']))

                self.text_objective_generator = TextObjectiveGenerator(self.grid_type, self.n_zone, sentence_file=sentence_file)

            # Number of objectives / frequency of change
            self.n_objectives = config["objective"]["curriculum"]["n_objective"]

            self.max_objectives = self.n_objectives + 10 # to avoid having 100 objective for bigger maze

            self.is_multi_objective = self.n_objectives > 1
            self.objective_changing_every = config["objective"]["curriculum"]["change_every"]

            # Construct the list of objectives (the order is always the same)
            self.all_objectives = list(itertools.product(range(self.n_row), range(self.n_col)))
            objective_shuffler = random.Random(777)
            objective_shuffler.shuffle(self.all_objectives)
            self.train_objectives = self.all_objectives[:self.n_objectives]
            self.test_objectives = self.all_objectives[self.n_objectives:self.max_objectives]

            self._reward_position = self.train_objectives[0]

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
        state["objective"] = self.get_objective()
        state["info"] = dict([('reward_position', self.reward_position),('agent_position', self.agent_position)])
        return state

    def get_current_objective(self):
        x,y = self._reward_position
        return self.grid[x,y]

    def get_objective(self):

        if self.current_objective is None:
            self.current_objective = self._create_objective()
#        self.current_objective = self._create_objective()

        return self.current_objective

    def _create_objective(self):

        if self.objective_type == 'text':
            objective = self._create_text_objective()
        else:
            objective = self._create_image_objective()
        return objective

    def _create_image_objective(self):

        x_rew, y_rew = self._reward_position
        label_id, color_obj = self.grid_label_color[x_rew, y_rew]['label'], self.grid_label_color[x_rew, y_rew]['color']

        if not self.use_background_for_objective:
            color_obj = '0_0_0'

        images_folder = self.path_to_objectives_images
        images_folder = images_folder.format(label_id, color_obj)
        objective, _ = load_image_or_fmap(path_to_images=images_folder)

        return objective

    def _create_text_objective(self):
        x,y = self._reward_position
        label, zone = self.grid_label_color[x,y]['label'], self.grid_label_color[x,y]['zone']
        text_objective = self.text_objective_generator.sample(label=label, zone=zone)
        return text_objective


    def _change_objective(self):

        # If objective has been used for enough epochs, change it.
        if self.count_current_objective >= self.objective_changing_every:
            self.reward_position = self.train_objectives[np.random.randint(len(self.train_objectives))]
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
            self.agent_position = (x, y)
            position_on_reward = bool(self.get_reward())
        return self.get_state()

    def step(self, action):
        current_x,current_y = copy(self.agent_position)
        reward = 0

        # Discourage invalid moves?
        wrong_action_penalty = 0.0

        if action == 0: # NORTH
            x_ = min(self.n_row-1, current_x+1)
            if x_ == current_x:
                reward -= wrong_action_penalty
            else:
                current_x = x_
        elif action == 1:  # SOUTH
            x_ = max(0, current_x-1)
            if x_ == current_x:
                reward -= wrong_action_penalty
            else:
                current_x = x_
        elif action == 2:  # WEST
            y_ = max(0, current_y-1)
            if y_ == current_y:
                reward -= wrong_action_penalty
            else:
                current_y = y_
        elif action == 3:  # EAST
            y_ = min(self.n_col-1, current_y+1)
            if y_ == current_y:
                reward -= wrong_action_penalty
            else:
                current_y = y_
        else:
            assert False, "Wrong action"

        self.agent_position = (copy(current_x), copy(current_y))
        observation = self.get_state()
        reward += self.get_reward()
        info = {'agent_position' : copy(self.agent_position), 'reward_position': copy(self._reward_position)}

        assert self.agent_position == (current_x, current_y), "Problem with agent position"

        if reward == 1:
            done = True
        else:
            done = False

        return observation, reward, done, info

    def get_current_square(self):
        x,y = self.agent_position
        return self.grid[x,y]

    def get_square(self, x, y):
        return self.grid[x,y]

    def get_current_square_and_all_directions(self):
        x,y = self.agent_position
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

        return np.concatenate(all_directions_obs, axis=0)

    def _get_reward_no_penalty(self):
        if np.all(self.agent_position == self._reward_position):
            return 1
        else:
            return 0

    def _get_reward_with_penalty(self):
        if np.all(self.agent_position == self._reward_position):
            return 1
        else:
            return -0.1

    def render(self, show=False):
        """
        This function print the board and the position of the agent
        ONLY in this function, the image format is (H,W,C)
        """

        custom_grid = np.copy(self.grid_plot)

        if self.agent_position != []:
            x,y = self.agent_position
            x_size, y_size = (28,28)
            x_middle = x_size//2
            y_middle = y_size//2

            x_rew, y_rew = self._reward_position

            # Display agent position as a red point.
            custom_grid[x,y, x_middle-3:x_middle+3, y_middle-3:y_middle+3, :] = [255,0,0]

            # Display reward position as a green point.
            custom_grid[x_rew,y_rew, x_middle-3:x_middle+3, y_middle-3:y_middle+3, :] = [0,255,0]

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

        default_background = np.ones((3, 10, 8)) * 255
        default_background[0, :, :] = np.tile(np.linspace(0, 255, 8),(10, 1))  # 8 and 10 are the max value for n_col and n_row
        default_background[2, :, :] = np.tile(np.linspace(255, 0, 10),(8, 1)).T  # 8 and 10 are the max value for n_col and n_row

        # Define your type of grid
        # Either sequential or per zone
        # The important part is self.grid_label_color
        if "sequential" in self.grid_type:

            if not self.use_background_for_state:
                default_background *= 0

            self.n_zone = 1
            # Create a grid where you indicated for each cell : it's color and the label associated
            # np_color_to_str is in image_utils (converting np.array to a string corresponding to RGB color)
            self.grid_label_color = np.array([[{'zone' : 0, 'color': default_background[:, j, i], 'label': j * self.n_col + i} for i in range(self.n_col)] for j in range(self.n_row)])


        # ======================= ZONE (TEXT MAZE) ========================
        # =================================================================
        elif "zone" in self.grid_type:

            zone_per_row = int(np.sqrt(self.n_zone))
            zone_per_col = int(np.sqrt(self.n_zone))

            row_per_zone = self.n_row
            col_per_zone = self.n_col

            tot_row = row_per_zone * zone_per_row
            tot_col = col_per_zone * zone_per_col



            if self.grid_type == "zone_color_gradient":

                self.grid_label_color = np.empty((tot_row, tot_col), dtype=object)
                for i in range(tot_row):
                    for j in range(tot_col):
                        self.grid_label_color[i,j] = dict()
                        self.grid_label_color[i, j]['color'] = default_background[:, i, j]
                        pos_x, pos_y = i % row_per_zone, j % col_per_zone
                        current_label = pos_x * self.n_col + pos_y
                        self.grid_label_color[i, j]['label'] = current_label

                        current_zone = i//row_per_zone*zone_per_col + j//col_per_zone
                        self.grid_label_color[i, j]['zone'] = current_zone
                        pass

                assert not np.any(self.grid_label_color==None), "Grid not full, there are some None values"

            else:

                self.grid_label_color = None

                count_zone = 0
                for i in range(zone_per_row):
                    bg_color = default_background[:, i*4, 0] # x4 is to have a bigger change of color

                    line = np.array([[{'zone': count_zone, 'color': bg_color, 'label': j * self.n_col + i} for i in range(self.n_col)] for j in range(self.n_row)])
                    count_zone += 1

                    for j in range(1,zone_per_col):
                        bg_color = default_background[:, i*4,j*4] # x4 is to have a bigger change of color

                        line = np.concatenate((line, np.array([[{'zone': count_zone, 'color': bg_color, 'label': j * self.n_col + i} for i in range(self.n_col)] for j in range(self.n_row)])), axis=1)
                        count_zone += 1


                    if self.grid_label_color is None:
                        self.grid_label_color = line
                    else:
                        self.grid_label_color = np.concatenate((self.grid_label_color, line),axis=0)

                assert count_zone == self.n_zone, "Problem in number of zone : count {}    self.n_zone {}".format(count_zone, self.n_zone)

            self.n_row = self.n_row * zone_per_row
            self.n_col = self.n_col * zone_per_col



        self.grid = np.zeros((self.n_row, self.n_col)+self.features_shape)

        # Based on what you indicate above, create the maze
        self.grid_plot = np.zeros((self.n_row, self.n_col, 28, 28, 3))
        for i in range(self.n_row):
            for j in range(self.n_col):

                current_label = self.grid_label_color[i,j]['label']
                color = np_color_to_str(self.grid_label_color[i,j]['color'])
                last_chosen_file = None

                if self.save_image:
                    path_to_image_raw = self.path_to_maze_images.format(current_label, color, 'raw')
                    self.grid_plot[i,j], last_chosen_file = load_image_or_fmap(path_to_images=path_to_image_raw)

                path_to_image = self.path_to_maze_images.format(current_label, color, self.features_type)
                img, _ = load_image_or_fmap(path_to_images=path_to_image,last_chosen_file=last_chosen_file)
                self.grid[i, j] = img
                assert not (np.isclose(self.grid[i,j], np.zeros(self.features_shape)).all()), "Grid is still zero, problem"

        if show :
            self.render()

    def eval(self):
        self.train_test_folder = 'test/'
    def train(self):
        self.train_test_folder = 'train/'

    def action_space(self):
        return 4


    def state_objective_dim(self):
        self.reset(show=False)
        state = self.get_state()
        state_objective_dim_dict = dict()
        state_objective_dim_dict['env_state'] = state['env_state'].shape

        if 'text' in self.objective_type:
            state_objective_dim_dict['objective'] = self.text_objective_generator.voc_size
        else:
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
