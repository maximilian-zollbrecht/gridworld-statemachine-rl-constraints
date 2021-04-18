from abc import abstractmethod
import numpy as np


# This represents a state_machine, that can be used to specify constraints
# the states are returned as numpy array with 1 as active state(s) and 0 as inactive states
# the first state with index 0 is always considered starting state
class StateMachine:

    # channel_shape refers to one positions-channel
    def __init__(self, num_states):
        self.num_states = num_states
        # the first state is always the start state
        self.state = 0
        self.last_state = 0

    # this should be called to tell the state machine about each steps input
    # here any logic to change the state can be implemented
    @abstractmethod
    def update(self, observation, action):
        raise NotImplementedError

    # returns the state as one hot array representation
    def get_states(self):
        states = np.zeros(self.num_states, dtype=int)
        states[self.state] = 1
        return states

    # returns, whether the state_machine is accepting the current input
    @abstractmethod
    def is_accepting(self):
        raise NotImplementedError

    def reset(self):
        self.last_state = self.state
        self.state = 0
