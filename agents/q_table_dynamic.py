import numpy


# Q-Table, that uses the states hash()-method to store it in a dict.
# Does not build full table at initialization, but instead generates entries during execution.
class QTable:
    def __init__(self, number_of_actions, initial_value=0.0, learning_rate=0.1, discount_factor=0.9):
        self.initial_value = initial_value
        self.number_of_actions = number_of_actions

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.q_table = dict()

        self.translation_file = open('D:\\Masterarbeit\\masterarbeit\\models\\qagent_translation.txt', "a")
        pass

    def get_best_action(self, state, available_actions=None):
        if available_actions is None:
            available_actions = range(self.number_of_actions)

        state_entry = self.get_state_info(state)
        # q-table entry filtered by available actions
        available_info = [state_entry[i] for i in available_actions]
        # print("--- Choosing new action:")
        # print("state: ", state.positions[0][0])
        # print("current information: ", state_entry)
        # print("actions legal: ", available_actions)
        # print("info from that: ", available_info)
        return numpy.argmax(available_info)

    def get_state_info(self, state):
        hashed_state = hash(state)
        if hashed_state in self.q_table:
            return self.q_table[hashed_state]
        else:
            return numpy.full(self.number_of_actions, self.initial_value)

    def update_state_info(self, state, new_info):
        hashed_state = hash(state)
        if hashed_state not in self.q_table:
            self.translation_file.write(str(hashed_state))
            self.translation_file.write("\n")
            self.translation_file.write(str(state))
            self.translation_file.write("\n")
            self.translation_file.flush()
        self.q_table[hashed_state] = new_info

    def action(self, state, action_index: int, new_state, reward):
        state_entry = self.get_state_info(state)
        new_state_entry = self.get_state_info(new_state)

        estimate = max(new_state_entry)

        # update q-value
        new_q = state_entry[action_index]\
            + self.learning_rate * (reward + self.discount_factor *
                                    estimate - state_entry[action_index])
        state_entry[action_index] = new_q

        # update q-table
        self.update_state_info(state, state_entry)

        # print("--- did an update on the q_table:")
        # print("from: ", state)
        # print("with action: ", action_index)
        # print("reward: ", reward)
        # print("new state: ", state_entry)
        pass

    def reset(self):
        self.q_table.clear()

    def __str__(self):
        s = ""
        for key in self.q_table:
            s += str(key) + ": " + str(self.q_table[key]) + "\n"
            # s += str(self.q_table[key]) + "\n"
        return s
