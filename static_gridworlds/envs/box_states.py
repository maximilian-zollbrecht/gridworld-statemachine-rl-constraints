from gym import register

import numpy as np

from static_gridworlds.gridworld import Gridworld
from static_gridworlds.gridworld_env import GridworldEnvironment
from static_gridworlds.gridworld_objects import Floor, Goal, Wall
from static_gridworlds.observation import StateMachineHistoryObservation, Observation
from static_gridworlds.state_machine import StateMachine


class BoxEnv(GridworldEnvironment):
    LEGAL_BOX_POSITIONS = [[1, 1], [2, 1], [2, 2]]

    name = 'Gridworld-Box-v1'

    @staticmethod
    def register():
        print('registering ' + BoxEnv.name)
        register(
            id=BoxEnv.name,
            entry_point='static_gridworlds.envs.box_states:BoxEnv',
            max_episode_steps=40,
        )

    def __init__(self):
        world = Gridworld([[Wall(), Wall(), Wall(), Wall(), Wall(), Wall()],
                           [Wall(), Floor(), Floor(), Wall(), Wall(), Wall()],
                           [Wall(), Floor(), Floor(), Floor(), Floor(), Wall()],
                           [Wall(), Wall(), Floor(), Floor(), Floor(), Wall()],
                           [Wall(), Wall(), Wall(), Floor(), Goal(), Wall()],
                           [Wall(), Wall(), Wall(), Wall(), Wall(), Wall()]],
                          [2, 1], False)

        self.box_position = [2, 2]
        self.state_machine = BoxStateMachine()

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
                if 0 <= behind_box[0] < len(self.world.grid[0]) and \
                   0 <= behind_box[1] < len(self.world.grid):
                    if isinstance(self.world.get_tile_at_position(behind_box), Floor):
                        self.box_position = behind_box
                        self.world.pos = new_agent_pos
                # if not: do nothing
            else:
                self.world.pos = new_agent_pos
        return

    def reset(self):
        self.box_position = [2, 2]
        self.state_machine.reset()
        super().reset()
        self.observation.add_states()

    # only give reward, when box is in correct position
    # def get_reward(self):
    #     if isinstance(self.world.get_current_position(), Goal):
    #         return 100 if self.box_position in BoxEnv.LEGAL_BOX_POSITIONS else -1
    #     else:
    #         return super().get_reward()

    # use the state machine to manipulate reward
    def get_tile_reward(self, tile, _):
        if isinstance(tile, Goal):
            return 30
        return -1

    def get_initial_observation(self, world):
        return StateMachineHistoryObservation(np.shape(self.world.grid),
                                              self.state_machine,
                                              static_channels=[self.world.get_observation_channel_walkable(),
                                                               self.world.get_observation_channel_lava(),
                                                               self.world.get_observation_channel_goal()]
                                              if world.include_static_layers else None, history=2)

    # # overwrite update_position to add box position as well
    def update_observation(self):
        super().update_observation()
        self.observation.add_position(self.world.get_observation_channel_position(self.box_position))

    def step(self, action_index):
        # move
        self.move_agent(action_index)
        # update observation, so state_machine can use it
        self.update_observation()
        # update state_machine
        self.state_machine.update(self.observation, action_index)
        self.observation.add_states()
        # use updated observation
        return self.observation, self.get_reward(action_index), self.is_over(), {}


class BoxEnvNoStates(BoxEnv):

    @staticmethod
    def register():
        print('registering ' + BoxEnv.name)
        register(
            id=BoxEnv.name,
            entry_point='static_gridworlds.envs.box_states:BoxEnvNoStates',
            max_episode_steps=40,
        )

    def __init__(self):
        super().__init__()

    def step(self, action_index):
        # move
        self.move_agent(action_index)
        # update state_machine
        self.state_machine.update(self.observation, action_index)
        # update observation
        self.update_observation()
        # use updated observation
        return self.observation, self.get_reward(action_index), self.is_over(), {}

    def get_initial_observation(self, world):
        return Observation(np.shape(self.world.grid),
                           static_channels=[self.world.get_observation_channel_walkable(),
                                            self.world.get_observation_channel_lava(),
                                            self.world.get_observation_channel_goal()]
                           if world.include_static_layers else None)


class BoxStateMachine(StateMachine):

    def __init__(self):
        super(BoxStateMachine, self).__init__(2)

    # State 0=box is at start
    # state 1=box is somewhere else
    def update(self, observation, action):
        box_position = observation.get_observation()[0]
        box_position = np.argmax(box_position)

        # skip if already in illegal state
        if self.state == 1:
            pass

        if not (box_position == 14 or box_position == 15):
            self.state = 1
        pass

    def is_accepting(self):
        return self.state == 0
