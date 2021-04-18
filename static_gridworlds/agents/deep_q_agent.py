import numpy as np

from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras import Model
from tensorflow.keras import layers

import tensorflow as tf

import csv

from static_gridworlds.agents.agent import StateAgent, update_reward
import random


# This Agent uses a deep-Q-network to learn
# A state_machine can be used to adjust reward and skip learning steps, but is not used as training input.
class DeepQAgent(StateAgent):

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

        self.logging = False
        pass

    # approximate Q function using Neural Network
    # observation (positions and state_machine) is input
    # Q Value of each action is output of network
    def build_model(self):
        input = layers.Input(shape=np.shape(self.env.observation.get_observation()), name="observation")
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

    # compile the model and log its shape and summary
    def compile_model(self):
        print("input_shape: ", np.shape(self.env.get_observation().get_observation()))
        print(self.model.summary())

        self.model.compile(loss=MeanSquaredError(), optimizer=Adam(lr=self.learning_rate))

    # Executes a single step in the underlying Gridworld_Env and learns from that step.
    def step(self):
        # store for learning
        cur_observation = self.env.get_observation().copy()

        # decide if random action or learned action
        if self.learning and random.random() < self.explore_rate:
            # select randomly
            action_ind = random.randrange(self.action_size)
        else:
            # prediction of q_values
            model_prediction = self.model.predict(np.array([cur_observation.get_observation()]))[0]
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
        if self.steps % self.train_cycle == self.train_cycle - 1 and self.learning:
            self.learn_from_history()

        self.steps += 1

        return reward, done

    # Adds a step to the replay buffer for later training
    def add_step_to_replay_buffer(self, current_observation, action_ind, next_observation, reward):
        self.observation_history.append(current_observation)
        self.action_history.append(action_ind)
        self.next_observation_history.append(next_observation)
        self.reward_history.append(reward)

    # maintain the replay buffer to make sure it doesnt grow above its wanted size
    def maintain_replay_buffer(self):
        # Limit the history
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

        # next observation
        next_positions = np.array([o.get_observation() for o in self.next_observation_history])

        # action results
        # predictions = np.array(self.prediction_history)
        actions = np.array(self.action_history)
        rewards = np.array(self.reward_history)

        # predict q-vaules for all observations in history
        predictions = self.model.predict(positions)

        # predict q-vaules for all next observations in history
        next_predictions = self.model.predict(next_positions)
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
        predicted_q_values = predictions * (1 - masks)

        q_result = q_values + predicted_q_values

        # for i in range(len(q_result)):
        #     print("----------")
        #     print(rewards[i])
        #     print(actions[i])
        #     print(predictions[i])
        #     print(q_result[i])

        # fit model with calculated values
        # print("---")
        # print(positions)
        # print(q_result)
        history = self.model.fit(positions,
                                 q_result,
                                 steps_per_epoch=min(self.batch_size,
                                                     len(self.action_history)),
                                 epochs=1, verbose=0)

        if self.logging:
            print("- learning step done")

        # store losses
        if self.log_file:
            self.loss_writer.writerow(history.history['loss'])

    # Also open file for loss-values
    def use_logfile(self):
        if self.log_file is not None:
            self.loss_file = open(self.log_file + "-loss.csv", self.file_mode)
            self.loss_writer = csv.writer(self.loss_file,
                                          quoting=csv.QUOTE_NONNUMERIC, delimiter=',', lineterminator='\n')
        super(DeepQAgent, self).use_logfile()

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

    # decrease explore rate after each training episode
    def train(self):
        steps, reward = super(DeepQAgent, self).train()
        # decrease explore_rate
        if self.explore_rate > self.explore_rate_min:
            self.explore_rate *= self.explore_rate_decay
        elif self.explore_rate > 0:
            self.explore_rate = self.explore_rate_min
        return steps, reward

    # def test(self):
    #     self.logging = True
    #     steps, reward = super(DeepQAgent, self).test()
    #     # self.logging = False
    #     return steps, reward

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
        return


# Tracks how often the agent received a negative reward greater than 5
class DeepQAgentIsland(DeepQAgent):

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
