import numpy

from datetime import datetime

numpy.set_printoptions(precision=3, suppress=True)


class EnvironmentTest:
    def __init__(self, agent, training_epochs, training_episodes, use_steps=False, log=False):
        self.agent = agent

        self.training_epochs = training_epochs
        self.training_episodes = training_episodes
        self.use_steps = use_steps
        self.log = log
        self.agent.reset()

    def run(self):
        t_start = datetime.now()
        t_end = None

        total_steps = 0
        epoch = 0

        if self.use_steps:
            while total_steps < self.training_epochs:
                epoch += 1
                total_steps += self.run_epoch(epoch, t_start)
            print("Trained for ", total_steps, " steps")
        else:
            for epoch in range(self.training_epochs):
                self.run_epoch(epoch, t_start)

        self.agent.finish_all()

    def run_epoch(self, epoch, t_start):
        epoch_steps = 0
        for i in range(self.training_episodes):
            t_end = datetime.now()
            steps, reward = self.agent.train()
            epoch_steps += steps
        agent_info = self.agent.finish_epoch()
        if self.log:
            print("----- epoch ended:", epoch)
            print("passed time:", t_end - t_start)
            # print("agent-info")
            print(agent_info)

        # test agent
        steps, reward = self.agent.test()
        # if self.log:
        #     print("Test done:")
        #     print("steps: ", steps)
        #     print("reward: ", reward)
        return epoch_steps

    def run_multiple(self, n):
        t_start = datetime.now()

        log_file = self.agent.log_file
        log_files = list()
        for i in range(n):
            self.agent.log_file = log_file + "-" + str(i)
            self.agent.reset()
            log_files.append(self.agent.log_file)
            self.run()
            t_end = datetime.now()
            print("Test #", i, " done")
            print("passed time in total:", t_end - t_start)

        return log_files
