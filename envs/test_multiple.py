import numpy
import gym

from static_gridworlds.agents.deep_q_agent_states import DeepQAgentStates
from static_gridworlds.agents.q_agent import QTableAgent
from static_gridworlds.envs.boat_race_states import BoatRaceEnv
from static_gridworlds.envs.box_states import BoxEnv
from static_gridworlds.envs.island_states import IslandEnv
from static_gridworlds.envs.test_env import EnvironmentTest

import static_gridworlds.plot_results as myplot


numpy.set_printoptions(precision=3, suppress=True)

IslandEnv.register()

env = gym.make(IslandEnv.name)
env.reset()


def reward_update(reward, is_accepting):
    if is_accepting:
        return reward
    return -1


agent = QTableAgent(env,
                    discount_factor=0.9,
                    learning_rate=1.0,
                    # explore_rate=1.1,
                    explore_rate=0.0,
                    explore_rate_decay=0.99,
                    explore_rate_min=0.0,
                    initial_value=-0.0,
                    log_file="D:\\Masterarbeit\\masterarbeit\\models\\island\\island-qtable-rewardrep2",
                    # ignore_on_wrong_state=True,
                    reward_update=reward_update,
                    # log_steps=True
                    )
# agent = DeepQAgentStates(env, discount_factor=0.9,
#                          learning_rate=0.0002,
#                          explore_rate=1.5,
#                          # explore_rate_decay=1,
#                          explore_rate_decay=0.98,
#                          explore_rate_min=0.2,
#                          # model_file='D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-base.h5',
#                          log_file="D:\\Masterarbeit\\masterarbeit\\models\\box\\box-dqn-ignore",
#                          train_cycle=50,
#                          batch_size=1000,
#                          ignore_on_wrong_state=True,
#                          # reward_update=reward_update,
#                          # log_steps=True,
#                          # file_mode="a+"
#                          )


log_files = EnvironmentTest(agent, 1000, 1, True).run_multiple(10)

print("logged into: ", log_files)

#myplot.plot(log_file="D:\\Masterarbeit\\masterarbeit\\models\\box\\box-qtable", episode_group=16)

myplot.plot_multiple(log_files, limit_steps=1000, plot_steps=50, plot_steps_test=50)

# myplot.plot_loss(log_files)

# myplot.plot_episode_reward_explore(log_files)
# myplot.plot_test_reward_accepting(log_files)
#
# myplot.plot_episode_reward_explore2(log_files)
# myplot.plot_test_reward_accepting2(log_files)
