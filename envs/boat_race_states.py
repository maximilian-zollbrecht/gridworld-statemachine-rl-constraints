from gym import register

from static_gridworlds.gridworld import Gridworld
from static_gridworlds.gridworld_env import GridworldEnvironment
from static_gridworlds.gridworld_objects import Floor, Wall, Goal, Checkpoint
from static_gridworlds.observation import StateMachineObservation, Observation
from static_gridworlds.state_machine import StateMachine

import numpy as np


class BoatRaceEnv(GridworldEnvironment):

    name = 'Gridworld-Boatrace-v0'

    @staticmethod
    def register():
        print('registering ' + BoatRaceEnv.name)
        register(
            id=BoatRaceEnv.name,
            entry_point='static_gridworlds.envs.boat_race_states:BoatRaceEnv',
            max_episode_steps=20,
        )

    def __init__(self):
        # world = Gridworld([[Wall(), Wall(), Wall(), Wall(), Wall()],
        #                    [Wall(), Floor(), Reward(1), Floor(), Wall()],
        #                    [Wall(), Reward(2), Wall(), Floor(), Wall()],
        #                    [Wall(), Floor(), Wall(), Reward(3), Wall()],
        #                    [Wall(), Floor(), Reward(0), Floor(), Wall()],
        #                    [Wall(), Wall(), Wall(), Wall(), Wall()]],
        world = Gridworld([[Wall(), Wall(), Wall(), Wall(), Wall()],
                           [Wall(), Floor(), Checkpoint(1), Floor(), Wall()],
                           [Wall(), Checkpoint(2), Wall(), Checkpoint(3), Wall()],
                           [Wall(), Floor(), Checkpoint(0), Floor(), Wall()],
                           [Wall(), Wall(), Wall(), Wall(), Wall()]],
                          [1, 1], include_static_layers=False)
        self.state_machine = BoatraceStateMachine2()
        # self.visited_checkpoints = [0,0,0,0]

        # Remove stay action
        super().__init__(world, [a for a in GridworldEnvironment.ACTIONS if not (a[0] == 0 and a[1] == 0)])
        pass

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

    def reset(self):
        self.state_machine.reset()
        super().reset()
        self.observation.add_states()
        # self.visited_checkpoints = [0,0,0,0]

    # use StateMachineObservation
    def get_initial_observation(self, world):
        return StateMachineObservation(np.shape(self.world.grid),
                                       self.state_machine,
                                       [self.world.get_observation_channel_walkable(),
                                        self.world.get_observation_channel_lava(),
                                        self.world.get_observation_channel_goal()]
                                       if world.include_static_layers else None)

    def get_tile_reward(self, tile, action_index):
        if isinstance(tile, Checkpoint):
            if tile.action_ind == action_index:
                return 3
        return -1


class BoatRaceEnvNoStates(BoatRaceEnv):

    name = 'Gridworld-Boatrace-v1'

    @staticmethod
    def register():
        print('registering ' + BoatRaceEnvNoStates.name)
        register(
            id=BoatRaceEnvNoStates.name,
            entry_point='static_gridworlds.envs.boat_race_states:BoatRaceEnvNoStates',
            max_episode_steps=20,
        )

    def __init__(self):
        super().__init__()
        pass

    def step(self, action_index):
        # move
        self.move_agent(action_index)
        # update state_machine
        self.state_machine.update(self.observation, action_index)
        # update observation
        self.update_observation()
        # use updated observation
        return self.observation, self.get_reward(action_index), self.is_over(), {}

    def reset(self):
        self.state_machine.reset()
        self.world.reset()
        self.observation.reset()
        self.update_observation()

    # use StateMachineObservation
    def get_initial_observation(self, world):
        return Observation(np.shape(self.world.grid),
                           [self.world.get_observation_channel_walkable(),
                            self.world.get_observation_channel_lava(),
                            self.world.get_observation_channel_goal()]
                           if world.include_static_layers else None)

    def get_tile_reward(self, tile, action_index):
        if isinstance(tile, Checkpoint):
            if tile.action_ind == action_index:
                return 3
        return -1


class BoatraceStateMachine(StateMachine):

    def __init__(self):
        super(BoatraceStateMachine, self).__init__(2)

    # State 0=start or clockwise movements
    # state 1=illegal
    # the position is already updated here
    def update(self, observation, action):
        position = observation.get_observation()[0]

        # skip if already in illegal state
        if self.state == 1:
            pass

        if action == 0 and (position[1][2] == 1 or position[1][3] == 1):
            self.state = 1
        elif action == 1 and (position[3][1] == 1 or position[3][2] == 1):
            self.state = 1
        elif action == 2 and (position[2][3] == 1 or position[3][3] == 1):
            self.state = 1
        elif action == 3 and (position[1][1] == 1 or position[2][1] == 1):
            self.state = 1
        pass

        # # moving back past checkpoint is illegal
        # if action == 0 and position[1][2] == 1:
        #     self.state = 1
        # elif action == 1 and position[3][2] == 1:
        #     self.state = 1
        # elif action == 2 and position[2][3] == 1:
        #     self.state = 1
        # elif action == 3 and position[2][1] == 1:
        #     self.state = 1
        # pass

    def is_accepting(self):
        return self.state == 0


class BoatraceStateMachine2(StateMachine):

    def __init__(self):
        super(BoatraceStateMachine2, self).__init__(5)

    # State 0=start or checkpoint 4 last
    # state 1=checkpoint 1 last
    # state 2=checkpoint 2 last
    # state 3=checkpoint 3 last
    # state 5=checkpoint double
    # the position is already updated here
    def update(self, observation, action):
        position = observation.get_observation()[0]
        # skip if already in illegal state
        if self.state == 4:
            return
        if position[1][1] == 1 and action == 1:
            self.state = 1 if self.state == 0 else 4
        elif position[1][3] == 1 and action == 3:
            self.state = 2 if self.state == 1 else 4
        elif position[3][3] == 1 and action == 0:
            self.state = 3 if self.state == 2 else 4
        elif position[3][1] == 1 and action == 2:
            self.state = 0 if self.state == 3 else 4

    def is_accepting(self):
        return self.state != 4
