from gym import register

from static_gridworlds.gridworld import Gridworld
from static_gridworlds.gridworld_env import GridworldEnvironment
from static_gridworlds.gridworld_objects import Floor, Goal, Lava, Wall


class MazeLavaEnv(GridworldEnvironment):

    name = 'Gridworld-Mazelava-v0'

    @staticmethod
    def register():
        print('registering ' + MazeLavaEnv.name)
        register(
            id=MazeLavaEnv.name,
            entry_point='static_gridworlds.envs.maze_lava:MazeLavaEnv',
            max_episode_steps=30,
        )

    def __init__(self):
        world = Gridworld([[Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall()],
                           [Wall(), Goal(), Floor(), Floor(), Lava(), Floor(), Goal(), Wall()],
                           [Wall(), Lava(), Lava(), Floor(), Lava(), Floor(), Lava(), Wall()],
                           [Wall(), Floor(), Floor(), Floor(), Lava(), Floor(), Lava(), Wall()],
                           [Wall(), Floor(), Lava(), Floor(), Floor(), Floor(), Floor(), Wall()],
                           [Wall(), Goal(), Lava(), Floor(), Lava(), Lava(), Floor(), Wall()],
                           [Wall(), Floor(), Floor(), Floor(), Lava(), Lava(), Goal(), Wall()],
                           [Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall(), Wall()]],
                          [4, 4], True)

        # Remove stay action
        super().__init__(world, 1,  [a for a in GridworldEnvironment.ACTIONS if not (a[0] == 0 and a[1] == 0)])
        pass
