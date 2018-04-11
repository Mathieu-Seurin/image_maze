import numpy as np
from itertools import count

class Maze(object):
    def __init__(self):

        self.cases=[]

        origin = Case(north=1, south=2, east=3, west=4)
        self.cases.append(origin)
        north = Case(north=None, south=0, east=None, west=None)
        self.cases.append(north)
        south = Case(north=0, south=None, east=None, west=None)
        self.cases.append(south)
        east = Case(north=None, south=None, east=None, west=0)
        self.cases.append(east)
        west = Case(north=None, south=None, east=0, west=None)
        self.cases.append(west)

        self.current_state = 0

    def step(self, action):
        if action == 0: # NORTH
            id_case = self.current_state.north()
        elif action == 1:  # SOUTH
            id_case = self.current_state.south()
        elif action == 2:  # EAST
            id_case = self.current_state.east()
        elif action == 3:  # WEST
            id_case =  self.current_state.west()
        else:
            assert False, "Wrong action"

        self.current_state = self.cases[id_case]

        return observation, 0, done, info

class Case(object):
    _ids = count(0)

    def __init__(self, current,  **kwargs):
        self.id = self._ids.next()
        self.directions = kwargs
    def north(self):
        return self.directions["north"]
    def south(self):
        return self.directions["south"]
    def east(self):
        return self.directions["east"]
    def west(self):
        return self.directions["west"]

class Image(object):
    def __init__(self):
        self.image_name = 0

    def get_formated_image(self):
        pass

    def get_raw_image(self):
        pass