from static_gridworlds.gridworld import Gridworld
from static_gridworlds.gridworld_env import GridworldEnvironment
from static_gridworlds.gridworld_objects import Floor, Goal, Lava
from gym.envs.registration import register


class LavaEnv(GridworldEnvironment):

    @staticmethod
    def register():
        name = 'Gridworld-Lava-v0'
        print('registering ' + name)
        register(
            id=name,
            entry_point='static_gridworlds.envs.lava:LavaEnv',
            max_episode_steps=50
        )

    def __init__(self):
        world = Gridworld([[Floor(), Floor(), Lava(), Lava(), Lava(), Floor(), Goal()],
                           [Floor(), Floor(), Floor(), Floor(), Floor(), Floor(), Floor()],
                           [Floor(), Floor(), Floor(), Floor(), Floor(), Floor(), Floor()],
                           [Floor(), Floor(), Floor(), Floor(), Floor(), Floor(), Floor()],
                           [Floor(), Floor(), Lava(), Lava(), Lava(), Floor(), Floor()]],
                          [0, 0])

        # Remove stay action
        super().__init__(world)
        pass
