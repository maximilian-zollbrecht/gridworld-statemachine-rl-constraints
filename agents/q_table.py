import numpy


# Old Q-Table, that tried to use the state directly to store values
class QTable:
    def __init__(self, actions, states, initial_value=0.0, learning_rate=0.1, discount_factor=0.9):
        self.actions = actions
        self.states = states
        self.initial_value = initial_value

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.q_table = numpy.full((len(states), len(actions)), initial_value)
        pass

    def get_best_action(self, state, available_actions=None):
        actions = self.actions if available_actions is None else available_actions
        state_info = self.get_state_info(state)

        action_indices = [self.actions.index(a) for a in actions]
        available_info = [state_info[i] for i in action_indices]

        return self.actions[action_indices[numpy.argmax(available_info)]]

    def get_state_info(self, state):
        state_ind = self.states.index(state)
        return self.q_table[state_ind, :]

    def action(self, state, action, new_state, reward):
        action_ind = self.actions.index(action)
        state_ind = self.states.index(state)
        new_state_ind = self.states.index(new_state)

        estimate = max(self.q_table[new_state_ind, :])

        new_q = (1 - self.learning_rate) * self.q_table[state_ind][action_ind]\
            + self.learning_rate * (reward + self.discount_factor * estimate)

        # print("Q-Table: going from ", state, " to ", new_state, "with action ", action, " got reward ", reward)
        # print("Q-Table: old_q ", self.q_table[state_ind][action_ind], " estimate ", estimate, " new_q", new_q)

        self.q_table[state_ind, action_ind] = new_q
        pass

    def __str__(self):
        s = " ".join(map(str, self.actions)) + "\n"
        for i in range(len(self.q_table)):
            s += str(self.states[i]) + str(self.q_table[i]) + "\n"

        return s
