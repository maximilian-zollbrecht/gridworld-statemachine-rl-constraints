import numpy
import gym

from static_gridworlds.agents.deep_q_agent_states import DeepQAgentStatesIsland
from static_gridworlds.agents.deep_q_agent import DeepQAgentIsland
from static_gridworlds.agents.q_agent import QTableAgentIsland

from static_gridworlds.envs.island_states import IslandEnv, IslandEnvNoStates
from static_gridworlds.envs.test_env import EnvironmentTest

import static_gridworlds.plot_results as plot


numpy.set_printoptions(precision=3, suppress=True)

IslandEnvNoStates.register()

env = gym.make(IslandEnvNoStates.name)
env.reset()


def reward_update(reward, is_accepting):
    if is_accepting:
        return reward
    return -20


# agent = QTableAgent(env,
#                     discount_factor=0.9,
#                     learning_rate=1.0,
#                     # explore_rate=1.1,
#                     explore_rate=1.0,
#                     explore_rate_decay=0.95,
#                     explore_rate_min=0.1,
#                     initial_value=100.0,
#                     log_file="D:\\Masterarbeit\\masterarbeit\\models\\island2\\island-nostates-qtable-ignore+100",
#                     ignore_on_wrong_state=True,
#                     # reward_update=reward_update,
#                     # log_steps=True
#                     )
agent = DeepQAgentIsland(env, discount_factor=0.9,
                         learning_rate=0.0002,
                         explore_rate=1.0,
                         # explore_rate_decay=1,
                         explore_rate_decay=0.95,
                         explore_rate_min=0.1,
                         # model_file='D:\\Masterarbeit\\masterarbeit\\models\\island2\\island-dqn-base.h5',
                         log_file="D:\\Masterarbeit\\masterarbeit\\models\\island2\\island-nostates-dqn-base",
                         train_cycle=5,
                         batch_size=100,
                         # ignore_on_wrong_state=True,
                         # reward_update=reward_update,
                         # log_steps=True,
                         # file_mode="a+"
                         )


log_files = EnvironmentTest(agent, 1000, 1, True, True).run_multiple(10)

print("logged into: ", log_files)

#myplot.plot(log_file="D:\\Masterarbeit\\masterarbeit\\models\\box\\box-qtable", episode_group=16)

plot.plot_multiple(log_files, limit_steps=1000, plot_steps=50, plot_steps_test=50)
print("Fell into water", agent.fell_into_water, "times")

# myplot.plot_loss(log_files)

# myplot.plot_episode_reward_explore(log_files)
# myplot.plot_test_reward_accepting(log_files)
#
# myplot.plot_episode_reward_explore2(log_files)
# myplot.plot_test_reward_accepting2(log_files)
