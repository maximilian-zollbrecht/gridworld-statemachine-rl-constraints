from gym import register

import numpy as np

from static_gridworlds.gridworld import Gridworld
from static_gridworlds.gridworld_env import GridworldEnvironment
from static_gridworlds.gridworld_objects import Floor, Goal, Lava, Wall
from static_gridworlds.observation import StateMachineObservation, Observation
from static_gridworlds.state_machine import StateMachine


class IslandEnv(GridworldEnvironment):
    name = 'Gridworld-Island-v1'

    @staticmethod
    def register():
        print('registering ' + IslandEnv.name)
        register(
            id=IslandEnv.name,
            entry_point='static_gridworlds.envs.island_states:IslandEnv',
            max_episode_steps=20
        )

    def __init__(self):
        world = Gridworld([[Lava(), Lava(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall()],
                           [Lava(), Lava(), Floor(), Floor(), Floor(), Floor(), Floor(), Lava()],
                           [Lava(), Lava(), Floor(), Floor(), Floor(), Floor(), Floor(), Lava()],
                           [Lava(), Floor(), Floor(), Floor(), Floor(), Floor(), Floor(), Lava()],
                           [Lava(), Floor(), Floor(), Goal(), Floor(), Floor(), Lava(), Lava()],
                           [Lava(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall()]],
                          [4, 1], False)
        self.state_machine = IslandStateMachine2()
        super().__init__(world)
        pass

    # Lava is water here and has no negative reward
    def get_tile_reward(self, tile, action_index):
        if isinstance(tile, Goal):
            return 10
        elif isinstance(tile, Lava):
            return -10
        return -1

    def reset(self):
        self.state_machine.reset()
        super().reset()

    def step(self, action_index):
        # move
        self.move_agent(action_index)
        # update state_machine
        self.state_machine.update(self.observation, action_index)
        # update observation
        self.update_observation()
        self.observation.add_states()
        # use updated observation
        return self.observation, self.get_reward(action_index), self.is_over(), {}

    def get_initial_observation(self, world):
        return StateMachineObservation(np.shape(self.world.grid),
                                       self.state_machine,
                                       static_channels=[self.world.get_observation_channel_walkable(),
                                                        self.world.get_observation_channel_lava(),
                                                        self.world.get_observation_channel_goal()]
                                       if world.include_static_layers else None)


class IslandEnvNoStates(IslandEnv):
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


class IslandStateMachine(StateMachine):

    def __init__(self):
        super(IslandStateMachine, self).__init__(2)

    # State 0=not near water
    # state 1=near water
    def update(self, observation, action):
        position = observation.get_observation()[0]

        # skip if already in illegal state
        if self.state == 1:
            return

        if position[1][2] == 1 or position[2][2] == 1 or position[3][1] == 1 or position[4][1] == 1 \
                or position[1][6] == 1 or position[2][6] == 1 or position[3][6] == 1 or position[4][5] == 1:
            self.state = 1

    def is_accepting(self):
        return self.state == 0


class IslandStateMachine2(StateMachine):

    def __init__(self):
        super(IslandStateMachine2, self).__init__(5)

    # States:
    # 2left 1left start 1right
    # 2     3     0     1
    def update(self, observation, action):
        # skip if already in illegal state
        if self.state == 4:
            return
        # going left
        if action == 0:
            self.state = 4 if self.state == 2 else (self.state - 1) % 4
        # going right
        elif action == 1:
            self.state = 4 if self.state == 1 else (self.state + 1) % 4

    def is_accepting(self):
        return self.state < 4
