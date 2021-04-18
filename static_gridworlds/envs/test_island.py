import numpy
import gym
import tensorflow.keras as keras

from static_gridworlds.agents.deep_q_agent_states import DeepQAgentStatesIsland
from static_gridworlds.agents.deep_q_agent import DeepQAgentIsland
from static_gridworlds.agents.q_agent import QTableAgentIsland

from static_gridworlds.envs.island_states import IslandEnv
from static_gridworlds.envs.test_env import EnvironmentTest

from static_gridworlds.plot_results import plot

numpy.set_printoptions(precision=3, suppress=True)

IslandEnv.register()

env = gym.make(IslandEnv.name)
env.reset()


def reward_update(reward, is_accepting):
    if is_accepting:
        return reward
    return reward-2


agent = QTableAgentIsland(env,
                    discount_factor=0.9,
                    learning_rate=1.0,
                    # explore_rate=1.1,
                    explore_rate=1.0,
                    explore_rate_decay=0.95,
                    explore_rate_min=0.1,
                    initial_value=100.0,
                    log_file="D:\\Masterarbeit\\masterarbeit\\saves\\tmp",
                    # ignore_on_wrong_state=True,
                    reward_update=reward_update,
                    # log_steps=True
                    )
# agent = DeepQAgentStates(env, discount_factor=0.9,
#                          learning_rate=0.0002,
#                          explore_rate=1.0,
#                          # explore_rate_decay=1,
#                          explore_rate_decay=0.95,
#                          explore_rate_min=0.1,
#                          # model_file='D:\\Masterarbeit\\masterarbeit\\saves\\island-dqn-rewardsub.h5',
#                          log_file="D:\\Masterarbeit\\masterarbeit\\saves\\tmp",
#                          train_cycle=5,
#                          batch_size=100,
#                          # ignore_on_wrong_state=True,
#                          reward_update=reward_update,
#                          # log_steps=True,
#                          # file_mode="a+"
#                          )

EnvironmentTest(agent, 40, 5).run()

plot(log_file="D:\\Masterarbeit\\masterarbeit\\saves\\tmp", episode_group=10)
print("Fell into water", agent.fell_into_water, "times")
