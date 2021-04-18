from collections import deque
import numpy as np


# This represents a observation in a static_gridworld environment
# The position is expected to be one nested array, that is the same structure for each step
# valid values are 0 and 1
# Most notably makes everything hashable to allow storing in qtable
class Observation:

    # channel_shape refers to one positions-channel
    def __init__(self, channel_shape, static_channels=None):
        self.channel_shape = channel_shape
        self.observation = np.array([np.zeros(channel_shape)]).astype('int8')
        if static_channels:
            self.observation = np.append(self.observation, static_channels, axis=0).astype('int8')

    # Adds a position to the observation
    def add_position(self, position):
        if self.channel_shape != np.shape(position):
            raise Exception("Invalid channel_shape for positions channel")
        self.observation[0] = position

    # resets the observation by filling the position-part with zeroed arrays
    def reset(self):
        self.observation[0] = np.zeros(self.channel_shape)

    # returns the current observation including positions and static channels
    def get_observation(self):
        return self.observation

    def __hash__(self):
        bin_number = int(''.join([str(num) for num in self.observation.flatten()]), 2)

        # bin_number = 0
        # for num in self.observation.flatten():
        #     bin_number = bin_number*2 + num
        return bin_number

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str(self.observation[0])

    def copy(self):
        # observation is set manually, so static_channels can be ignored
        o = Observation(self.channel_shape)
        o.observation = self.observation.copy()
        return o


# This represents a history observation in a static_gridworld environment
# The observation is expected to be one or multiple nested arrays, that are the same structure for each channel
class HistoryObservation(Observation):

    # channel_shape refers to one positions-channel
    def __init__(self, channel_shape, static_channels=None, history=1):
        self.channel_shape = channel_shape
        self.observation = static_channels
        self.history = history

        if self.observation:
            for i in range(history):
                self.observation = np.append([np.zeros(channel_shape)], self.observation, axis=0)
        else:
            self.observation = np.array([np.zeros(channel_shape)])
            for i in range(history-1):
                self.observation = np.append([np.zeros(channel_shape)], self.observation, axis=0)

        self.observation = self.observation.astype('int8')

    # Adds a position to the observation
    def add_position(self, position):
        if self.channel_shape != np.shape(position):
            raise Exception("Invalid channel_shape for positions channel")
        # i.e. history=3, new=n
        # [1,2,3,s1,s2,s3]
        # [n,1,2,s1,s2,s3]
        for i in range(self.history-1):
            self.observation[i+1] = self.observation[i]
        self.observation[0] = position

    # resets the observation by filling the position-part with zeroed arrays
    def reset(self):
        for i in range(self.history):
            self.observation[i] = np.zeros(self.channel_shape)

    def copy(self):
        o = HistoryObservation(self.channel_shape, history=self.history)
        o.observation = self.observation.copy()
        return o

    def __str__(self):
        return str(self.observation[:self.history])


# This represents a observation in a static_gridworld environment with a state_machine
# The observation is expected to be one or multiple nested arrays, that are the same structure for each channel
# and one additional array for the state_machine
class StateMachineHistoryObservation(HistoryObservation):

    def __init__(self, channel_shape, state_machine, static_channels=None, history=1):
        super().__init__(channel_shape, static_channels=static_channels, history=history)

        # add state_machine reference
        self.state_machine = state_machine
        self.states = state_machine.get_states()

    def __hash__(self):
        bin_str = ''.join([str(num) for num in np.append(self.observation.flatten(), self.states)])
        bin_number = int(bin_str, 2)

        # bin_number = 0
        # for num in self.observation.flatten():
        #     bin_number = bin_number*2 + num
        return bin_number

    def __str__(self):
        s = super(StateMachineHistoryObservation, self).__str__()
        s += str(self.states)
        return s

    def add_states(self):
        self.states = self.state_machine.get_states()

    def copy(self):
        # observation is set manually, so static_channels can be ignored
        o = StateMachineHistoryObservation(self.channel_shape, self.state_machine, history=self.history)
        o.observation = self.observation.copy()
        return o


# This represents a observation in a static_gridworld environment with a state_machine
# The observation is expected to be one or multiple nested arrays, that are the same structure for each channel
# and one additional array for the state_machine
class StateMachineObservation(Observation):

    def __init__(self, channel_shape, state_machine, static_channels=None):
        super().__init__(channel_shape, static_channels=static_channels)

        # add state_machine reference
        self.state_machine = state_machine
        self.states = state_machine.get_states()

    def __hash__(self):
        bin_str = ''.join([str(num) for num in np.append(self.observation.flatten(), self.states)])
        bin_number = int(bin_str, 2)

        # bin_number = 0
        # for num in self.observation.flatten():
        #     bin_number = bin_number*2 + num
        return bin_number

    def __str__(self):
        s = super(StateMachineObservation, self).__str__()
        s += str(self.states)
        return s

    def add_states(self):
        self.states = self.state_machine.get_states()

    def copy(self):
        # observation is set manually, so static_channels can be ignored
        o = StateMachineObservation(self.channel_shape, self.state_machine)
        o.observation = self.observation.copy()
        o.states = self.states
        return o

# worldgrid = np.array([[0, 1, 0],
#                       [1, 0, 1],
#                       [0, 1, 0]])
# worldshape = np.shape(worldgrid)
#
# # two static channels
# obs = Observation(worldshape, [np.zeros(worldshape), np.ones(worldshape)])
# print("-----")
# print(obs.get_observation())
#
# obs.add_position(worldgrid)
# worldgrid[1][1]=2
# obs.add_position(worldgrid)
# print("-----")
# print(obs.get_observation())
#
#
# print(np.shape(obs.get_observation()))

