from abc import abstractmethod
import csv


# Represents an Agent in a Gridworld_Env.
# During training and test its step() method is called repeatedly.
# This method has to be implemented by the overriding class, using a learning algorithm.
# Handles test and training and logs results to files.
class Agent:
    def __init__(self, env, log_file=None, log_steps=False, file_mode="w"):
        self.env = env
        self.learning = True
        self.file_mode = file_mode
        self.log_file = log_file
        self.log_steps = log_steps
        self.log_file = log_file
        self.use_logfile()

        self.episode_file = None
        self.episode_writer = None
        self.test_file = None
        self.test_writer = None

        self.episode_step_file = None
        self.episode_step_writer = None
        self.test_step_file = None
        self.test_step_writer = None
        pass

    # tells the agent, that an epoch has ended
    def finish_epoch(self):
        return "nothing here"

    # tells the agent, that the program has ended
    def finish_all(self):
        return "nothing here"

    # resets the agent
    def reset(self):
        self.env.reset()
        pass

    # open csv writers again with currently set log_file
    def use_logfile(self):
        if self.log_file is not None:
            self.episode_file = open(self.log_file + "-episode.csv", self.file_mode)
            self.episode_writer = csv.writer(self.episode_file,
                                             quoting=csv.QUOTE_NONNUMERIC, delimiter=',', lineterminator='\n')
            self.test_file = open(self.log_file + "-test.csv", self.file_mode)
            self.test_writer = csv.writer(self.test_file,
                                          quoting=csv.QUOTE_NONNUMERIC, delimiter=',', lineterminator='\n')
            if self.log_steps:
                self.episode_step_file = open(self.log_file + "-step.csv", self.file_mode)
                self.episode_step_writer = csv.writer(self.episode_step_file,
                                                      quoting=csv.QUOTE_NONNUMERIC, delimiter=',', lineterminator='\n')
                self.test_step_file = open(self.log_file + "-test-step.csv", self.file_mode)
                self.test_step_writer = csv.writer(self.test_step_file,
                                                   quoting=csv.QUOTE_NONNUMERIC, delimiter=',', lineterminator='\n')

    # executes one step in the environment
    @abstractmethod
    def step(self):
        return 0, True

    # define, what should be logged after a single step in test episode
    def log_teststep(self, reward):
        return [reward]

    # define, what should be logged after a single step in training episode
    def log_episodestep(self, reward):
        return [reward]

    # define, what should be logged after a test episode
    def log_test(self, steps, reward):
        return [steps, reward]

    # define, what should be logged after a training episode
    def log_training(self, steps, reward):
        return [steps, reward]

    # executes one test episode
    def test(self):
        # stop learning during test
        self.learning = False

        steps, test_reward = self.run_episode(self.test_step_file, self.test_step_writer)

        if self.log_file:
            row = self.log_test(steps, test_reward)
            self.test_writer.writerow(row)

        self.env.reset()

        return steps, test_reward

    # executes one training episode
    def train(self):
        # start learning during training
        self.learning = True

        steps, episode_reward = self.run_episode(self.episode_step_file, self.episode_step_writer)

        if self.log_file:
            row = self.log_training(steps, episode_reward)
            self.episode_writer.writerow(row)

        self.env.reset()

        return steps, episode_reward

    # Lets this agent execute for one episode
    # returns reward and number of steps in this episode
    def run_episode(self, step_file, step_writer):
        done = False
        episode_reward = 0
        steps = 0

        # move until done once
        while not done:
            step_reward, done = self.step()
            episode_reward += step_reward
            steps += 1
            # Log step if enabled
            if self.log_steps:
                step_writer.writerow(self.log_episodestep(step_reward))

        # Flush to files
        if self.log_file is not None:
            self.episode_file.flush()
            if self.log_steps:
                step_file.flush()

        return steps, episode_reward


# This method is used to adjust the reward of the agent according to the state_machine's result
# Default is to return reward unchanged
def update_reward(reward, _):
    return reward


# Adds Options to handle state machine results
class StateAgent(Agent):

    def __init__(self, env, ignore_on_wrong_state=False, reward_update=update_reward, log_file=None, log_steps=False, file_mode="w"):
        self.ignore_on_wrong_state = ignore_on_wrong_state
        self.reward_update = reward_update
        super().__init__(env, log_file=log_file, log_steps=log_steps, file_mode=file_mode)

    # Updates the reward using the Agent's specified function
    def get_reward(self, reward, is_accepting):
        return self.reward_update(reward, is_accepting)

    # executes one step in the environment
    @abstractmethod
    def step(self):
        return 0, True
