from static_gridworlds.gridworld_objects import Lava, Goal, Checkpoint, Floor
import numpy as np


# Represents a Gridworld
# holds agent position
# provides channels walkable, lava and goal
class Gridworld:

    def __init__(self, grid, pos=None, include_static_layers=True):
        # the grid is used to store non-movable objects
        self.grid = np.array(grid)
        self.include_static_layers = include_static_layers

        self.walkable = self.get_empty_grid()
        self.lava = self.get_empty_grid()
        self.goal = self.get_empty_grid()

        self.width = len(self.grid[0])
        self.height = len(self.grid)

        for y in range(self.height):
            for x in range(self.width):
                tile = self.grid[y][x]
                self.walkable[y][x] = 1 if tile.walkable() else 0
                self.lava[y][x] = 1 if isinstance(tile, Lava) else 0
                self.goal[y][x] = 1 if isinstance(tile, Goal) or isinstance(tile, Checkpoint) else 0

        self.pos = None
        self.starting_pos = pos
        self.reset()
        return

    # returns the shape of the gridworld
    def get_shape(self):
        return np.shape(self.grid)

    # Returns the agents current position
    def get_current_position(self):
        return self.get_tile_at_position(self.pos)

    # Returns the Tile at the given position
    def get_tile_at_position(self, position):
        return self.grid[position[1]][position[0]]

    # Returns the tile the agent is currently on
    def get_agent_tile(self):
        return self.grid[self.pos[1]][self.pos[0]]

    # Returns if a given postion can be entered by the agent
    def is_walkable(self, position):
        return self.walkable[position[1]][position[0]] == 1

    # # returns the current positions
    # # if with_static_layers is true static layers will be added to the agents position
    # # old, and currently unused
    # def compute_observation(self):
    #     if self.with_static_layers:
    #         return [self.get_observation_channel_position(self.pos),
    #                 self.get_observation_channel_walkable(),
    #                 self.get_observation_channel_lava(),
    #                 self.get_observation_channel_goal()]
    #     else:
    #         return [self.get_observation_channel_position(self.pos)]

    # Gets an positions channel with walkable tiles marked
    def get_observation_channel_walkable(self):
        return self.walkable

    # Gets an positions channel with lava tiles marked
    def get_observation_channel_lava(self):
        return self.lava

    # Gets an positions channel with goal tiles marked
    def get_observation_channel_goal(self):
        return self.goal

    # Gets an positions channel that is zeroed except for the agents position
    def get_observation_channel_position(self, position):
        channel = self.get_empty_grid()
        channel[position[1]][position[0]] = 1
        return channel

    # returns a empty grid with the worlds size
    def get_empty_grid(self):
        return np.zeros(self.grid.shape)

    # resets the world
    def reset(self):
        # reset agent position
        if callable(self.starting_pos):
            self.pos = self.starting_pos()
            if not (isinstance(self.pos, list) and len(self.pos) == 2):
                raise TypeError("did expect a callable returning a 2-size array or a 2-size array as position")
        elif isinstance(self.starting_pos, list) and len(self.starting_pos) == 2:
            self.pos = self.starting_pos
        else:
            raise TypeError("did expect a callable returning a 2-size array or a 2-size array as position")
        pass

