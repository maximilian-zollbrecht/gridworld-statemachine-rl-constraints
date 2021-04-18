from abc import abstractmethod

import gym

from gym import spaces

import numpy as np
from static_gridworlds.observation import Observation
from static_gridworlds.gridworld import Gridworld
from static_gridworlds.gridworld_objects import Goal, Lava, Checkpoint


# Integrates Gridworld and openai Gym.
# Implements agent movement and environment reset.
class GridworldEnvironment(gym.Env):
    ACTIONS = [
        # left
        [-1, 0],
        # right
        [1, 0],
        # up
        [0, -1],
        # down
        [0, 1],
        # stay
        [0, 0]
    ]

    def __init__(self, world, actions=None):
        if not isinstance(world, Gridworld):
            raise TypeError("world needs to be a Gridworld")
        self.world = world

        # add 3 static channels if set
        self.observation = self.get_initial_observation(world)

        self.actions = GridworldEnvironment.ACTIONS if actions is None else actions
        self.action_indices = [i for i in range(len(self.actions))]
        self.action_space = spaces.Discrete(len(self.actions))
        pass

    def get_initial_observation(self, world):
        return Observation(np.shape(self.world.grid),
                           [self.world.get_observation_channel_walkable(),
                            self.world.get_observation_channel_lava(),
                            self.world.get_observation_channel_goal()] if world.include_static_layers else None)

    @staticmethod
    @abstractmethod
    def register():
        raise NotImplementedError

    def step(self, action_index):
        self.move_agent(action_index)
        self.update_observation()

        # print("Moving with action: ", action_index, " now at ", self.world.pos)
        return self.get_observation(), self.get_reward(action_index), self.is_over(), {}

    def update_observation(self):
        self.observation.add_position(self.world.get_observation_channel_position(self.world.pos))

    def get_observation(self):
        return self.observation

    # Always the same world... do nothing
    def seed(self, seed=None):
        return seed

    def reset(self):
        self.world.reset()
        self.observation.reset()
        self.update_observation()

    # Not implemented
    def render(self, mode='human', close=False):
        pass

    # Moves the agent by the action of the given index
    # if the move is illegal, nothing is done instead
    def move_agent(self, action_index):
        if self.is_legal(action_index, self.world.pos):
            action = self.actions[action_index]
            self.world.pos = [self.world.pos[0] + action[0], self.world.pos[1] + action[1]]
        pass

    def is_legal(self, action_index, position):
        action = self.actions[action_index]
        new_pos = [position[0] + action[0], position[1] + action[1]]
        return (0 <= new_pos[0] < len(self.world.grid[0])
                and 0 <= new_pos[1] < len(self.world.grid)
                and self.world.is_walkable(new_pos))

    def get_current_actions(self):
        return [action_index for action_index in range(self.actions.__len__()) if self.is_legal(action_index, self.world.pos)]

    # currently unused
    def get_actions(self, position):
        return [action_index for action_index in range(self.actions.__len__()) if self.is_legal(action_index, position)]

    def get_reward(self, action_index):
        return self.get_tile_reward(self.world.get_agent_tile(), action_index)

    def is_over(self):
        return self.world.get_current_position().end()

    def get_tile_reward(self, tile, action_index):
        if isinstance(tile, Goal):
            return 100
        elif isinstance(tile, Lava):
            return -10
        elif isinstance(tile, Checkpoint):
            if tile.action_ind == action_index:
                return 20
        return -1
