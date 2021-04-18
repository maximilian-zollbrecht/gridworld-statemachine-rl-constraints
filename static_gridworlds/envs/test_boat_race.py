import numpy
import gym

from static_gridworlds.agents.deep_q_agent import DeepQAgent
from static_gridworlds.agents.deep_q_agent_states import DeepQAgentStates
from static_gridworlds.agents.q_agent import QTableAgent
from static_gridworlds.envs.boat_race_states import BoatRaceEnv, BoatRaceEnvNoStates
from static_gridworlds.envs.test_env import EnvironmentTest

from static_gridworlds.plot_results import plot

numpy.set_printoptions(precision=3, suppress=True)

BoatRaceEnv.register()

env = gym.make(BoatRaceEnv.name)
env.reset()


def reward_update(reward, is_accepting):
    if is_accepting:
        return reward
    return reward-2


# agent = QTableAgent(env,
#                     discount_factor=0.9,
#                     learning_rate=1.0,
#                     explore_rate=1.1,
#                     explore_rate_decay=0.97,
#                     explore_rate_min=0.1,
#                     initial_value=-0.0,
#                     log_file="D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-temp",
#                     # ignore_on_wrong_state=True,
#                     reward_update=reward_update,
#                     # log_steps=True
#                     )
# EnvironmentTest(agent, 40, 2).run()
# plot(log_file="D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-qtable-temp", episode_group=8)
agent = DeepQAgentStates(env, discount_factor=0.9,
                   learning_rate=0.0002,
                   explore_rate=1.1,
                   # explore_rate_decay=1,
                   explore_rate_decay=0.96,
                   explore_rate_min=0.1,
                   # model_file='D:\\Masterarbeit\\masterarbeit\\models\\boatrace-dqn-rewardsub.h5',
                   log_file="D:\\Masterarbeit\\masterarbeit\\saves\\tmp",
                   train_cycle=30,
                   batch_size=300,
                   # ignore_on_wrong_state=True,
                   reward_update=reward_update,
                   # log_steps=True,
                   # file_mode="a+"
                   )
EnvironmentTest(agent, 40, 5, log=True).run()
plot(log_file="D:\\Masterarbeit\\masterarbeit\\saves\\tmp", episode_group=8)
