import numpy
import gym

from static_gridworlds.agents.deep_q_agent_states import DeepQAgentStates, DeepQAgentStatesUnique
from static_gridworlds.agents.q_agent import QTableAgent
from static_gridworlds.envs.box_states import BoxEnv, BoxEnvNoStates
from static_gridworlds.envs.test_env import EnvironmentTest

from static_gridworlds.plot_results import plot

numpy.set_printoptions(precision=3, suppress=True)

BoxEnv.register()

env = gym.make(BoxEnv.name)
env.reset()


def reward_update(reward, is_accepting):
    if is_accepting:
        return reward
    return reward-5


# agent = QTableAgent(env,
#                     discount_factor=0.9,
#                     learning_rate=1.0,
#                     explore_rate=2.0,
#                     explore_rate_decay=0.995,
#                     explore_rate_min=0.2,
#                     initial_value=0.0,
#                     log_file="D:\\Masterarbeit\\masterarbeit\\models\\box\\box-nostates-qtable-rewardsub2",
#                     # ignore_on_wrong_state=True,
#                     reward_update=reward_update,
#                     # log_steps=True
#                     )
agent = DeepQAgentStates(env, discount_factor=0.9,
                         learning_rate=0.0002,
                         explore_rate=1.5,
                         # explore_rate_decay=1,
                         explore_rate_decay=0.995,
                         explore_rate_min=0.2,
                         model_file='D:\\Masterarbeit\\masterarbeit\\models\\box-dqn-rewardsub.h5',
                         log_file="D:\\Masterarbeit\\masterarbeit\\models\\tmp",
                         train_cycle=50,
                         batch_size=1000,
                         # ignore_on_wrong_state=True,
                         reward_update=reward_update,
                         # log_steps=True,
                         # file_mode="a+"
                         )
EnvironmentTest(agent, 10000, 4, True, True).run()
plot(log_file="D:\\Masterarbeit\\masterarbeit\\models\\tmp", episode_group=8)
