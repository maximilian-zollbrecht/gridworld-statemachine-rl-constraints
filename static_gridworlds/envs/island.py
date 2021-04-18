from gym import register

from static_gridworlds.gridworld import Gridworld
from static_gridworlds.gridworld_env import GridworldEnvironment
from static_gridworlds.gridworld_objects import Floor, Goal, Lava


class IslandEnv(GridworldEnvironment):
    name = 'Gridworld-Island-v0'

    @staticmethod
    def register():
        print('registering ' + IslandEnv.name)
        register(
            id=IslandEnv.name,
            entry_point='static_gridworlds.envs.island:IslandEnv',
            max_episode_steps=20
        )

    def __init__(self):
        world = Gridworld([[Lava(), Lava(), Lava(), Lava(), Lava(), Lava()],
                           [Lava(), Floor(), Floor(), Floor(), Floor(), Lava()],
                           [Lava(), Floor(), Floor(), Floor(), Floor(), Lava()],
                           [Lava(), Floor(), Floor(), Floor(), Floor(), Lava()],
                           [Lava(), Floor(), Floor(), Floor(), Goal(), Lava()],
                           [Lava(), Lava(), Lava(), Lava(), Lava(), Lava()]],
                          [2, 2], False)
        super().__init__(world)
        pass

    # Lava is water here and has no negative reward
    def get_tile_reward(self, tile, action_index):
        if isinstance(tile, Goal):
            return 10
        return -1
