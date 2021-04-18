from gym import register

from static_gridworlds.gridworld import Gridworld
from static_gridworlds.gridworld_env import GridworldEnvironment
from static_gridworlds.gridworld_objects import Floor, Wall, Goal, Checkpoint


class BoatRaceEnv(GridworldEnvironment):

    name = 'Gridworld-Boatrace-v0'

    @staticmethod
    def register():
        print('registering ' + BoatRaceEnv.name)
        register(
            id=BoatRaceEnv.name,
            entry_point='static_gridworlds.envs.boat_race:BoatRaceEnv',
            max_episode_steps=20,
        )

    def __init__(self):
        world = Gridworld([[Wall(), Wall(), Wall(), Wall(), Wall()],
                           [Wall(), Floor(), Checkpoint(1), Floor(), Wall()],
                           [Wall(), Checkpoint(2), Wall(), Checkpoint(3), Wall()],
                           [Wall(), Floor(), Checkpoint(0), Floor(), Wall()],
                           [Wall(), Wall(), Wall(), Wall(), Wall()]],
                          [1, 1])
                          # lambda: [random.randint(0, 2), random.randint(0, 2)])

        self.visited_checkpoints = list()

        # Remove stay action
        super().__init__(world, 2, [a for a in GridworldEnvironment.ACTIONS if not (a[0] == 0 and a[1] == 0)])
        pass

    # remember visited checkpoints
    def get_reward(self, action_index):
        reward = super().get_reward(action_index)
        if reward > 0:
            checkpoint = 1 if self.world.pos == [2,1] else (2 if self.world.pos == [3,2] else (3 if self.world.pos == [2,3] else 4))
            self.visited_checkpoints.append(checkpoint)
        return reward
