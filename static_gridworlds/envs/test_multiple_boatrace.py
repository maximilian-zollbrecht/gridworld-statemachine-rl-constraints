import numpy
import gym

from static_gridworlds.agents.deep_q_agent import DeepQAgent
from static_gridworlds.agents.deep_q_agent_states import DeepQAgentStates
from static_gridworlds.agents.q_agent import QTableAgent
from static_gridworlds.envs.boat_race_states import BoatRaceEnv, BoatRaceEnvNoStates
from static_gridworlds.envs.test_env import EnvironmentTest

import static_gridworlds.plot_results as myplot


numpy.set_printoptions(precision=3, suppress=True)

BoatRaceEnvNoStates.register()

env = gym.make(BoatRaceEnvNoStates.name)
env.reset()


def reward_update(reward, is_accepting):
    if is_accepting:
        return reward
    return -10


agent = QTableAgent(env,
                    discount_factor=0.9,
                    learning_rate=1.0,
                    explore_rate=1.1,
                    explore_rate_decay=0.96,
                    explore_rate_min=0.2,
                    initial_value=0.0,
                    log_file="D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-qtable-nostates",
                    # ignore_on_wrong_state=True,
                    reward_update=reward_update,
                    # log_steps=True
                    )
# agent = DeepQAgent(env, discount_factor=0.9,
#                    learning_rate=0.0002,
#                    explore_rate=1.1,
#                    explore_rate_decay=0.96,
#                    explore_rate_min=0.1,
#                    log_file="D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-nostates",
#                    train_cycle=30,
#                    batch_size=300,
#                    ignore_on_wrong_state=True,
#                    reward_update=reward_update,
#                    # log_steps=True,
#                    # file_mode="a+"
#                    )


log_files = EnvironmentTest(agent, 200, 1, False, True).run_multiple(10)

print("logged into: ", log_files)

# myplot.plot_multiple(log_files, limit_steps=4000, plot_steps=100, plot_steps_test=100)

# myplot.plot_loss(log_files)

# myplot.plot_episode_reward_explore(log_files)
# myplot.plot_test_reward_accepting(log_files)
#
myplot.plot_episode_reward_explore2(log_files)
myplot.plot_test_reward_accepting2(log_files)
