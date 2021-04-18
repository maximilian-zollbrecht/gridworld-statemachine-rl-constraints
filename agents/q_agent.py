import random

from static_gridworlds.agents.agent import StateAgent, update_reward
from static_gridworlds.agents.q_table_dynamic import QTable


# This Agent uses a Q-table to learn
# A state_machine can be used to adjust reward and skip learning steps.
# It is used as training input, if included in the Observation's hash() method.
class QTableAgent(StateAgent):

    def __init__(self, env, discount_factor=0.9,
                 learning_rate=0.1,
                 explore_rate=1,
                 explore_rate_decay=0.9995,
                 explore_rate_min=0.05,
                 initial_value=0.0,
                 ignore_on_wrong_state=False,
                 reward_update=update_reward,
                 log_file="D:\\Masterarbeit\\masterarbeit\\models\\default-qtable",
                 log_steps=False
                 ):
        super().__init__(env, ignore_on_wrong_state=ignore_on_wrong_state,
                         reward_update=reward_update, log_file=log_file, log_steps=log_steps)

        # exploration vs. exploitation
        self.explore_rate_start = explore_rate
        self.explore_rate = explore_rate
        self.explore_rate_decay = explore_rate_decay
        self.explore_rate_min = explore_rate_min

        self.q_table = QTable(len(env.actions),
                              learning_rate=learning_rate,
                              discount_factor=discount_factor,
                              initial_value=initial_value)
        self.action_size = len(self.env.actions)
        self.accepting = True

        self.steps = 0

        self.learning = True
        self.logging = False
        pass

    def take_action(self):
        if self.learning and random.uniform(0, 1) < self.explore_rate:
            action = random.randint(0, self.action_size-1)
        else:
            action = self.q_table.get_best_action(self.env.get_observation(), self.env.action_indices)
        return action

    def __str__(self):
        return str(self.q_table)

    def step(self):
        # preserve current observation for next learning step
        observation = self.env.get_observation().copy()

        # take step
        action_ind = self.take_action()
        new_observation, reward, done, info = self.env.step(action_ind)
        # check if state_machine is accepting the resulting state
        self.accepting = self.env.state_machine.is_accepting()
        # update reward, according to given strategy
        reward = self.get_reward(reward, self.accepting)

        # log if needed
        if self.logging:
            print("action: ", action_ind)
            if self.accepting:
                print("self.accepting")

        # apply data to q-table
        if self.learning and not (self.ignore_on_wrong_state and not self.accepting):
            # if reward > 0:
            #     print("got reward")
            self.q_table.action(observation, action_ind, new_observation, reward)

        self.steps += 1

        return reward, done

    def train(self):
        steps, reward = super(QTableAgent, self).train()
        # decrease explore_rate
        if self.explore_rate > self.explore_rate_min:
            self.explore_rate *= self.explore_rate_decay
        elif self.explore_rate > 0:
            self.explore_rate = self.explore_rate_min
        return steps, reward

    # def test(self):
    #     self.logging = True
    #     steps, reward = super(QTableAgent, self).test()
    #     self.logging = False
    #     return steps, reward

    def finish_epoch(self):
        # flush log_files
        self.episode_file.flush()
        self.test_file.flush()

        s = "explore_rate: " + str(self.explore_rate)
        return s

    def finish_all(self):
        self.episode_file.close()
        self.test_file.close()

        print(self.q_table)

    def log_episodestep(self, reward):
        return [reward, self.accepting, self.explore_rate]

    def log_test(self, steps, reward):
        return [steps, reward, self.accepting, self.explore_rate]

    def log_teststep(self, reward):
        return [reward, self.accepting, self.explore_rate]

    def log_training(self, steps, reward):
        return [steps, reward, self.accepting, self.explore_rate]

    def reset(self):
        self.q_table.reset()
        self.action_size = len(self.env.actions)
        self.explore_rate = self.explore_rate_start
        self.accepting = True
        self.steps = 0
        self.use_logfile()


# Tracks how often the agent received a negative reward greater than 5
class QTableAgentIsland(QTableAgent):

    def __init__(self, env, discount_factor=0.9, learning_rate=0.1, explore_rate=1, explore_rate_decay=0.9995,
                 explore_rate_min=0.05, initial_value=0.0, ignore_on_wrong_state=False, reward_update=update_reward,
                 log_file="D:\\Masterarbeit\\masterarbeit\\models\\default-qtable", log_steps=False):
        super().__init__(env, discount_factor, learning_rate, explore_rate, explore_rate_decay, explore_rate_min,
                         initial_value, ignore_on_wrong_state, reward_update, log_file, log_steps)
        self.fell_into_water = 0

    # track result before overwriting reward
    def get_reward(self, reward, is_accepting):
        # track if the agent fell into the water during training
        if self.learning and reward < -5:
            self.fell_into_water += 1
        return super().get_reward(reward, is_accepting)

    # Do not reset fell_into_water. It should be measured over all runs.
