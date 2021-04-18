from gym import register

from static_gridworlds.gridworld import Gridworld
from static_gridworlds.gridworld_env import GridworldEnvironment
from static_gridworlds.gridworld_objects import Floor, Goal, Lava


class SmallLavaEnv(GridworldEnvironment):

    name = 'Gridworld-Smalllava-v0'

    @staticmethod
    def register():
        print('registering ' + SmallLavaEnv.name)
        register(
            id=SmallLavaEnv.name,
            entry_point='static_gridworlds.envs.small_lava:SmallLavaEnv',
            max_episode_steps=30,
        )

    def __init__(self):
        world = Gridworld([[Goal(), Lava(), Lava(), Lava(), Goal()],
                           [Floor(), Floor(), Floor(), Floor(), Floor()],
                           [Floor(), Floor(), Floor(), Floor(), Floor()],
                           [Floor(), Floor(), Floor(), Floor(), Floor()],
                           [Goal(), Lava(), Lava(), Lava(), Goal()]],
                          [2, 2], True)

        # Remove stay action
        super().__init__(world)
        pass
