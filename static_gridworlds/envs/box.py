from gym import register

import numpy as np

from static_gridworlds.gridworld import Gridworld
from static_gridworlds.gridworld_env import GridworldEnvironment
from static_gridworlds.gridworld_objects import Floor, Goal, Wall
from static_gridworlds.observation import HistoryObservation


class BoxEnv(GridworldEnvironment):
    LEGAL_BOX_POSITIONS = [[1, 1], [2, 1], [2, 2]]

    name = 'Gridworld-Box-v0'

    @staticmethod
    def register():
        print('registering ' + BoxEnv.name)
        register(
            id=BoxEnv.name,
            entry_point='static_gridworlds.envs.box:BoxEnv',
            max_episode_steps=30,
        )

    def __init__(self):
        world = Gridworld([[Wall(), Wall(), Wall(), Wall(), Wall(), Wall()],
                           [Wall(), Floor(), Floor(), Wall(), Wall(), Wall()],
                           [Wall(), Floor(), Floor(), Floor(), Floor(), Wall()],
                           [Wall(), Wall(), Floor(), Floor(), Floor(), Wall()],
                           [Wall(), Wall(), Wall(), Floor(), Goal(), Wall()],
                           [Wall(), Wall(), Wall(), Wall(), Wall(), Wall()]],
                          [1, 1], False)

        self.box_position = [2, 2]

        super().__init__(world)
        pass

    def move_agent(self, action_index):
        if self.is_legal(action_index, self.world.pos):
            action = self.actions[action_index]
            new_agent_pos = [self.world.pos[0] + action[0], self.world.pos[1] + action[1]]

            # Move box, if agent is pushing it
            if new_agent_pos == self.box_position:
                behind_box = [new_agent_pos[0] + action[0], new_agent_pos[1] + action[1]]
                # check if box can be moved
                if 0 <= behind_box[0] < len(self.world.grid[0]) and 0 <= behind_box[1] < len(self.world.grid):
                    if isinstance(self.world.get_tile_at_position(behind_box), Floor):
                        self.box_position = behind_box
                        self.world.pos = new_agent_pos
                # if not: do nothing
            else:
                self.world.pos = new_agent_pos
        return

    def reset(self):
        super().reset()
        self.box_position = [1, 1]

    # only give reward, when box is in correct position
    # def get_reward(self):
    #     if isinstance(self.world.get_current_position(), Goal):
    #         return 100 if self.box_position in BoxEnv.LEGAL_BOX_POSITIONS else -1
    #     else:
    #         return super().get_reward()

    def get_initial_observation(self, world):
        return HistoryObservation(np.shape(self.world.grid),
                                  static_channels=[self.world.get_observation_channel_walkable(),
                                                   self.world.get_observation_channel_lava(),
                                                   self.world.get_observation_channel_goal()]
                                  if world.include_static_layers else None,
                                  history=2)

    # overwrite update_position to add box position as well
    def update_observation(self):
        super().update_observation()
        self.observation.add_position(self.world.get_observation_channel_position(self.box_position))
