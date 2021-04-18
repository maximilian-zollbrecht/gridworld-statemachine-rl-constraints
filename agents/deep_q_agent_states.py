import numpy as np

from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras import Model
from tensorflow.keras import layers

import tensorflow as tf

import matplotlib.pyplot as plt

import csv

from static_gridworlds.agents.agent import StateAgent, update_reward
import random


class DeepQAgentStates(StateAgent):

    def __init__(self, env, discount_factor=0.8,
                 learning_rate=0.001,
                 explore_rate=1,
                 explore_rate_decay=0.9995,
                 explore_rate_min=0.02,
                 model_file=None,
                 train_cycle=10,
                 batch_size=100,
                 ignore_on_wrong_state=False,
                 reward_update=update_reward,
                 log_file="D:\\Masterarbeit\\masterarbeit\\models\\default-dqn",
                 log_steps=False,
                 file_mode="w"):
        super().__init__(env, ignore_on_wrong_state=ignore_on_wrong_state,
                         reward_update=reward_update, log_file=log_file, log_steps=log_steps, file_mode=file_mode)

        print("using shape", np.shape(self.env.get_observation().get_observation()))

        self.action_size = len(self.env.actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.last_action = None

        # exploration vs. exploitation
        self.explore_rate_start = explore_rate
        self.explore_rate = explore_rate
        self.explore_rate_decay = explore_rate_decay
        self.explore_rate_min = explore_rate_min

        # prepare model
        self.model_file = model_file
        self.model = self.build_model()
        self.compile_model()

        # tracking progress
        self.accepting = True
        self.steps = 0

        # controls experience replay
        self.train_cycle = train_cycle
        self.batch_size = batch_size

        # used for experience replay
        self.observation_history = []
        self.prediction_history = []
        self.action_history = []
        self.next_observation_history = []
        self.reward_history = []
        self.next_states_history = []
        self.states_history = []

        self.logging = False
        pass

    # approximate Q function using Neural Network
    # observation (positions and state_machine) is input
    # Q Value of each action is output of network
    def build_model(self):
        input = layers.Input(shape=np.shape(self.env.observation.get_observation()), name="observation")
        states = layers.Input(shape=np.shape(self.env.state_machine.get_states()), name="states")
        conv1 = layers.Conv2D(filters=8,
                              kernel_size=3,
                              strides=1,
                              activation="relu",
                              data_format="channels_first",
                              padding="valid")(input)
        conv2 = layers.Conv2D(filters=16,
                              kernel_size=3,
                              strides=1,
                              activation="relu",
                              data_format="channels_first",
                              padding="valid")(conv1)
        flatten = layers.Flatten()(conv2)
        combined = layers.concatenate([flatten, states])
        dense1 = layers.Dense(32, activation='relu')(combined)
        dense2 = layers.Dense(16, activation='relu')(dense1)

        qs = layers.Dense(self.action_size, activation='linear', name='actor')(dense2)

        model = Model(inputs=[input, states], outputs=qs)
        try:
            if self.model_file:
                model.load_weights(self.model_file)
                print("model weights loaded")
            else:
                print("no model file specified")
        except OSError:
            print("error: no model weights found")

        return model

    def compile_model(self):
        print("input_shape: ", np.shape(self.env.get_observation().get_observation()))
        print(self.model.summary())

        self.model.compile(loss=MeanSquaredError(), optimizer=Adam(lr=self.learning_rate))

    def step(self):
        # store for learning
        cur_observation = self.env.get_observation().copy()

        # decide if random action or learned action
        if self.learning and random.random() < self.explore_rate:
            # select randomly
            action_ind = random.randrange(self.action_size)
        else:
            # prediction of q_values
            model_prediction = self.model.predict({
                "observation": np.array([cur_observation.get_observation()]),
                "states": np.array([self.env.state_machine.get_states()])
            })[0]
            # select best predicted action
            action_ind = np.argmax(model_prediction)

        # Actually execute the action
        next_observation, reward, done, info = self.env.step(action_ind)
        # check if state_machine is accepting
        self.accepting = self.env.state_machine.is_accepting()
        # update reward according to given strategy
        reward = self.get_reward(reward, self.accepting)

        if self.logging:
            print("Took action ", action_ind)

        # Prepare to learn step later
        if self.learning and not (self.ignore_on_wrong_state and not self.accepting):
            # if reward > 0:
            #     print("got reward")
            # store everything in history
            self.add_step_to_replay_buffer(cur_observation, action_ind, next_observation.copy(), reward)

        # learn
        if self.steps % self.train_cycle == self.train_cycle-1 and self.learning:
            self.learn_from_history()

        self.steps += 1

        return reward, done
    
    def add_step_to_replay_buffer(self, current_observation, action_ind, next_observation, reward):
        self.observation_history.append(current_observation)
        self.action_history.append(action_ind)
        self.next_observation_history.append(next_observation)
        self.reward_history.append(reward)

    def maintain_replay_buffer(self):
        # Limit the state and reward history
        size = len(self.action_history)

        # remove old values
        if size > self.batch_size:
            del self.observation_history[:size - self.batch_size]
            del self.action_history[:size - self.batch_size]
            del self.next_observation_history[:size - self.batch_size]
            del self.reward_history[:size - self.batch_size]

    # lets the agent learn the steps in its replay buffer
    def learn_from_history(self):
        self.maintain_replay_buffer()

        if len(self.action_history) == 0:
            return

        # store in numpy-arrays for further computation
        positions = np.array([o.get_observation() for o in self.observation_history])
        states = np.array([o.states for o in self.observation_history])

        # next observation
        next_positions = np.array([o.get_observation() for o in self.next_observation_history])
        next_states = np.array([o.states for o in self.next_observation_history])

        # action results
        # predictions = np.array(self.prediction_history)
        actions = np.array(self.action_history)
        rewards = np.array(self.reward_history)

        # predict q-vaules for all observations in history
        predictions = self.model.predict({
            "observation": positions,
            "states": states
        })

        # predict q-vaules for all next observations in history
        next_predictions = self.model.predict({
            "observation": next_positions,
            "states": next_states
        })
        # extract q_values for chosen action
        # current_preds = np.choose(actions, predictions.T)

        # Calculate new q_values
        # Learning rate is 1 here, since the neural net also uses one
        # and in this example no statistical rewards are given
        # q_values = current_preds + (rewards + self.discount_factor * tf.reduce_max(next_predictions, axis=1) - current_preds)
        q_values = rewards + \
            self.discount_factor * tf.reduce_max(next_predictions, axis=1)

        # print("-----")
        # print("reward: ", rewards[0])
        # print("new q: ", q_values[0])
        # print("old_q: ", np.choose(actions, predictions.T))
        # print("action: ", actions[0])
        # print("obs: ", obs[0])
        # print("states: ", states[0])

        # mask q_values, so only chosen action is used for loss
        masks = tf.one_hot(actions, self.action_size)
        q_values = q_values[:, None] * masks
        # use predictions everywhere else
        predicted_q_values = predictions * (1-masks)

        q_result = q_values + predicted_q_values

        # for i in range(len(q_result)):
        #     print("----------")
        #     print(rewards[i])
        #     print(actions[i])
        #     print(predictions[i])
        #     print(q_result[i])

        # fit model with calculated values
        history = self.model.fit({"observation": positions, "states": states},
                                 q_result,
                                 steps_per_epoch=min(self.batch_size,
                                                     len(self.action_history)),
                                 epochs=1, verbose=0)

        if self.logging:
            print("- learning step done")

        # if obs[0][0][3][3] == 1:
        # # if history.history['loss'][0] > 0 and rewards[0] > 0:
        #     print("-----")
        #     print("reward: ", rewards[0])
        #     print("new q: ", q_result[0])
        #     print("old_q: ", predictions[0])
        #     print("action: ", actions[0])
        #     print("obs: ", obs[0])
        #     print("states: ", states[0])

        # store losses
        if self.log_file:
            self.loss_writer.writerow(history.history['loss'])

    # Also open file for loss-values
    def use_logfile(self):
        if self.log_file is not None:
            self.loss_file = open(self.log_file + "-loss.csv", self.file_mode)
            self.loss_writer = csv.writer(self.loss_file,
                                          quoting=csv.QUOTE_NONNUMERIC, delimiter=',', lineterminator='\n')
        super(DeepQAgentStates, self).use_logfile()

    # reset everything this agent has learned/done
    # files written will be overwritten, if log_file isn't changed before calling this
    def reset(self):
        self.model = self.build_model()
        self.compile_model()

        self.observation_history = []
        self.prediction_history = []
        self.action_history = []
        self.next_observation_history = []
        self.reward_history = []

        self.action_size = len(self.env.actions)
        self.explore_rate = self.explore_rate_start
        self.accepting = True
        self.steps = 0
        self.use_logfile()

    def train(self):
        steps, reward = super(DeepQAgentStates, self).train()
        # decrease explore_rate
        if self.explore_rate > self.explore_rate_min:
            self.explore_rate *= self.explore_rate_decay
        elif self.explore_rate > 0:
            self.explore_rate = self.explore_rate_min
        return steps, reward

    def log_training(self, steps, reward):
        return [steps, reward, self.accepting, self.explore_rate]

    def log_test(self, steps, reward):
        return [steps, reward, self.accepting, self.explore_rate]

    def log_episodestep(self, reward):
        return [reward, self.accepting, self.explore_rate]

    def log_teststep(self, reward):
        return [reward, self.accepting, self.explore_rate]

    def finish_epoch(self):
        s = "explore_rate: " + str(self.explore_rate)

        # flush losses
        self.loss_file.flush()

        # Save data into files
        if self.model_file:
            # save model weights
            self.model.save_weights(self.model_file, overwrite=True)
            print("model weights saved")
        return s

    def finish_all(self):
        # stop learning
        self.learning = False

        self.episode_file.close()
        self.test_file.close()
        self.loss_file.close()

        # plot decisions of current model
        shape = self.env.world.get_shape()

        Qs = [[[self.get_pred_for_pos([y, x], state)
                for x in range(shape[0])]
               for y in range(shape[1])]
              for state in range(self.env.state_machine.num_states)]

        # Scale
        max_q = max(np.array(Qs).flatten().max(), -np.array(Qs).flatten().min()) * 1.8
        Qs = Qs / max_q

        # nullen = [[0 for x in range(shape[0])] for y in range(shape[1])]
        #
        # x_range = range(shape[0])
        # y_range = range(shape[1])
        #
        # links = [[-abs(Qs[y][x][0]) for x in x_range] for y in y_range]
        # links_farben = [[10 if Qs[y][x][0] > 0 else -10 for x in x_range] for y in y_range]
        #
        # rechts = [[abs(Qs[y][x][1]) for x in x_range] for y in y_range]
        # rechts_farben = [[10 if Qs[y][x][1] > 0 else -10 for x in x_range] for y in y_range]
        #
        # oben = [[abs(Qs[y][x][2]) for x in x_range] for y in y_range]
        # oben_farben = [[10 if Qs[y][x][2] > 0 else -10 for x in x_range] for y in y_range]
        #
        # unten = [[-abs(Qs[y][x][3]) for x in x_range] for y in y_range]
        # unten_farben = [[10 if Qs[y][x][3] > 0 else -10 for x in x_range] for y in y_range]
        #
        # plt.gca().invert_yaxis()
        # plt.quiver(x_range, y_range, links, nullen, links_farben, scale_units='xy', scale=1, pivot='tail', width=0.004)
        # plt.quiver(x_range, y_range, rechts, nullen, rechts_farben, scale_units='xy', scale=1, pivot='tail', width=0.004)
        # plt.quiver(x_range, y_range, nullen, oben, oben_farben, scale_units='xy', scale=1, pivot='tail', width=0.004)
        # plt.quiver(x_range, y_range, nullen, unten, unten_farben, scale_units='xy', scale=1, pivot='tail', width=0.004)
        #
        # plt.show()

        print("gewählte Aktionen:")
        for qs in Qs:
            print(np.rot90(np.fliplr(np.argmax(qs, axis=2))))
        return

    # used to calculate actions for displaying agents policy after training is done
    def get_pred_for_pos(self, pos, state):
        tile = self.env.world.get_tile_at_position(pos)
        self.env.state_machine.state = state
        states = self.env.state_machine.get_states()
        if tile.walkable() and not tile.end():
            self.env.get_observation().add_position(self.env.world.get_observation_channel_position(pos))
            return self.model.predict({
                "observation": np.array([self.env.get_observation().get_observation()]),
                "states": np.array([states])})[0]
        else:
            return np.zeros(self.action_size)


class DeepQAgentStatesUnique(DeepQAgentStates):

    def __init__(self, env, discount_factor=0.8, learning_rate=0.001, explore_rate=1, explore_rate_decay=0.9995,
                 explore_rate_min=0.02, model_file=None, train_cycle=10, ignore_on_wrong_state=False,
                 reward_update=update_reward, log_file="D:\\Masterarbeit\\masterarbeit\\models\\default-dqn",
                 log_steps=False):
        self.known_states = dict()
        self.replay_buffer_index = 0
        super().__init__(env, discount_factor, learning_rate, explore_rate, explore_rate_decay, explore_rate_min,
                         model_file, train_cycle, 9999, ignore_on_wrong_state, reward_update, log_file, log_steps)

    # only adds every state action pair once
    def add_step_to_replay_buffer(self, current_observation, action_ind, next_observation, reward):
        # only add unknown states

        # state not existing at all
        if hash(current_observation) not in self.known_states:
            print("added ", hash(current_observation))
            new_entry = dict()
            self.known_states[hash(current_observation)] = new_entry
        # at this point the state has to be existing
        entry = self.known_states[hash(current_observation)]
        # only add if action is unknown
        if action_ind not in entry:
            entry[action_ind] = self.replay_buffer_index
            print("added ", action_ind, "for ", hash(current_observation), " with index ", self.replay_buffer_index)
            super(DeepQAgentStatesUnique, self).add_step_to_replay_buffer(
                current_observation, action_ind, next_observation, reward)
            self.replay_buffer_index += 1
        # else update predictions
        else:
            replay_buffer_index = entry[action_ind]
            # only prediction should change
            # self.prediction_history[replay_buffer_index] = prediction

            # if self.observation_history[replay_buffer_index] != current_observation:
            #     print("something went wrong, obs")
            # if self.action_history[replay_buffer_index] != action_ind:
            #     print("something went wrong, action")
            # if self.next_observation_history[replay_buffer_index] != next_observation:
            #     print("something went wrong, next")
            # if self.reward_history[replay_buffer_index] != reward:
            #     print("something went wrong, reward")

    # dont do anything here. replay buffer is supposed to be
    def maintain_replay_buffer(self):
        pass

    def finish_all(self):
        super(DeepQAgentStatesUnique, self).finish_all()
        print(len(self.known_states))
        for entry in self.known_states:
            print(entry, self.known_states[entry])


# Tracks how often the agent received a negative reward greater than 5
class DeepQAgentStatesIsland(DeepQAgentStates):

    def __init__(self, env, discount_factor=0.8, learning_rate=0.001, explore_rate=1, explore_rate_decay=0.9995,
                 explore_rate_min=0.02, model_file=None, train_cycle=10, batch_size=100, ignore_on_wrong_state=False,
                 reward_update=update_reward, log_file="D:\\Masterarbeit\\masterarbeit\\models\\default-dqn",
                 log_steps=False, file_mode="w"):
        super().__init__(env, discount_factor, learning_rate, explore_rate, explore_rate_decay, explore_rate_min,
                         model_file, train_cycle, batch_size, ignore_on_wrong_state, reward_update, log_file, log_steps,
                         file_mode)
        self.fell_into_water = 0

    # Use less kernels in second conv2d layer.
    # World has more fields to make up for it.
    def build_model(self):
        input = layers.Input(shape=np.shape(self.env.observation.get_observation()), name="observation")
        conv1 = layers.Conv2D(filters=8,
                              kernel_size=3,
                              strides=1,
                              activation="relu",
                              data_format="channels_first",
                              padding="valid")(input)
        conv2 = layers.Conv2D(filters=8,
                              kernel_size=3,
                              strides=1,
                              activation="relu",
                              data_format="channels_first",
                              padding="valid")(conv1)
        flatten = layers.Flatten()(conv2)
        dense1 = layers.Dense(32, activation='relu')(flatten)
        dense2 = layers.Dense(16, activation='relu')(dense1)

        qs = layers.Dense(self.action_size, activation='linear', name='actor')(dense2)

        model = Model(inputs=input, outputs=qs)
        try:
            if self.model_file:
                model.load_weights(self.model_file)
                print("model weights loaded")
            else:
                print("no model file specified")
        except OSError:
            print("error: no model weights found")

        return model

    # track result before overwriting reward
    def get_reward(self, reward, is_accepting):
        # track if the agent fell into the water during training
        if self.learning and reward < -5:
            self.fell_into_water += 1
        return super().get_reward(reward, is_accepting)

    # Do not reset fell_into_water. It should be measured over all runs.

