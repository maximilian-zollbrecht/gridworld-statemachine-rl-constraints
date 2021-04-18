import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from collections import deque


# Colors for plots
color_reward = 'tab:blue'
color_explore = 'tab:orange'
color_rewardperstep = 'tab:cyan'
color_accepting = 'tab:green'
color_loss = 'tab:gray'


# Plot various averages for logs during test and training.
# Plots over steps instead of episodes.
# The granularity of the plot can be configured using plot_steps and plot_steps_test.
def plot_multiple(log_files=[], limit_steps=None, plot_steps=100, plot_steps_test=20):
    num_logs = len(log_files)

    max_steps = 0
    all_loss = list()
    all_episode_rewards = list()
    all_episode_explore_rate = list()
    all_episode_accepting = list()
    all_episode_accepting_total = list()
    all_episode_steps = list()

    all_test_rewards = list()
    all_test_accepting = list()

    for i in range(num_logs):
        log_file = log_files[i]

        try:
            loss_file = open(log_file + "-loss.csv")
            cur_loss = np.genfromtxt(loss_file, dtype=None, delimiter=",")
            all_loss.append(cur_loss)
        except FileNotFoundError:
            print("file not found. not plotting loss.")
        episode_file = open(log_file + "-episode.csv")
        episode = np.genfromtxt(episode_file, dtype=None, delimiter=",")
        test_file = open(log_file + "-test.csv")
        test = np.genfromtxt(test_file, dtype=None, delimiter=",")

        accepting_episodes = 0
        steps = 0
        episode_steps = list()
        episode_accepting = list()
        episode_rewards = list()
        episode_explore_rate = list()

        # Extract episode data
        for line in episode:
            steps += line[0]
            episode_steps.append(steps)
            episode_rewards.append(line[1])
            accepting_episodes += not line[2]
            episode_accepting.append(line[2])
            episode_explore_rate.append(line[3])
        max_steps = max(max_steps, steps)

        print(accepting_episodes, " Episoden ohne Verletzung der Anforderung in Durchlauf ", i)
        all_episode_accepting_total.append(accepting_episodes)

        all_episode_steps.append(episode_steps)
        all_episode_rewards.append(episode_rewards)
        all_episode_explore_rate.append(episode_explore_rate)
        all_episode_accepting.append(episode_accepting)

        accepting_tests = 0
        test_rewards = list()
        test_accepting = list()

        # extract test data
        for line in test:
            test_rewards.append(line[1])
            accepting_tests += line[2]
            test_accepting.append(line[2])

        all_test_rewards.append(test_rewards)
        all_test_accepting.append(test_accepting)

    if limit_steps is not None:
        max_steps = limit_steps

    episode_length = len(max(all_episode_steps, key=len))
    test_length = len(max(all_test_rewards, key=len))

    episodes_per_test = episode_length/test_length

    plot_points = [(i+1)*max_steps/plot_steps for i in range(plot_steps)]

    ####################################################
    #####               Training                   #####
    ####################################################

    # scale data
    e_r_avg, e_r_std = scale_episodes_to_steps(num_logs=num_logs,  plot_steps=plot_steps, plot_points=plot_points,
                                               steps=all_episode_steps, values=all_episode_rewards)
    e_a_avg, e_a_std = scale_episodes_to_steps(num_logs=num_logs, plot_steps=plot_steps, plot_points=plot_points,
                                               steps=all_episode_steps, values=all_episode_accepting)
    e_e_avg, e_e_std = scale_episodes_to_steps(num_logs=num_logs, plot_steps=plot_steps, plot_points=plot_points,
                                               steps=all_episode_steps, values=all_episode_explore_rate)

    # Prepare Plot
    fig, explorerate_axis = plt.subplots()
    explorerate_axis.set_xlabel('Schritte')
    explorerate_axis.set_ylabel('Erkundungsrate', color=color_explore)
    explorerate_axis.tick_params(axis='y', labelcolor=color_explore)

    reward_axis = explorerate_axis.twinx()  # instantiate a second axes that shares the same x-axis
    reward_axis.set_ylabel('Belohnung', color=color_reward)  # we already handled the x-label with ax1
    reward_axis.tick_params(axis='y', labelcolor=color_reward)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Plot Data
    avgs, stds, stds2 = get_avg_std_for_plot(plot_steps=plot_steps, plot_points=plot_points, averages=e_e_avg, stds=e_e_std)
    explorerate_axis.plot(plot_points, avgs, color=color_explore)
    # Varianz zwischen den Durchläufen
    explorerate_axis.fill_between(plot_points, avgs - stds2 / 2, avgs + stds2 / 2, color=color_explore, alpha=0.15)
    avgs, stds, stds2 = get_avg_std_for_plot(plot_steps=plot_steps, plot_points=plot_points, averages=e_r_avg, stds=e_r_std)
    reward_axis.plot(plot_points, avgs, color=color_reward)
    reward_axis.fill_between(plot_points, avgs - stds / 2, avgs + stds / 2, color=color_reward, alpha=0.15)
    plt.savefig(log_files[0] + "-episode_reward_explore.png")
    plt.show()

    # Prepare Plot
    fig, explorerate_axis = plt.subplots()
    explorerate_axis.set_xlabel('Schritte')
    explorerate_axis.set_ylabel('Erkundungsrate', color=color_explore)
    explorerate_axis.tick_params(axis='y', labelcolor=color_explore)

    accepting_axis = explorerate_axis.twinx()  # instantiate a second axes that shares the same x-axis
    accepting_axis.set_ylabel('akzeptierender Zustandsautomat', color=color_accepting)  # we already handled the x-label with ax1
    accepting_axis.tick_params(axis='y', labelcolor=color_accepting)
    accepting_axis.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
    accepting_axis.set_ylim([-0.02, 1.02])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Plot Data
    avgs, stds, stds2 = get_avg_std_for_plot(plot_steps=plot_steps, plot_points=plot_points, averages=e_e_avg,
                                             stds=e_e_std)
    explorerate_axis.plot(plot_points, avgs, color=color_explore)
    # Varianz zwischen den Durchläufen
    explorerate_axis.fill_between(plot_points, avgs - stds2 / 2, avgs + stds2 / 2, color=color_explore, alpha=0.15)
    avgs, stds, stds2 = get_avg_std_for_plot(plot_steps=plot_steps, plot_points=plot_points, averages=e_a_avg,
                                             stds=e_a_std)
    # for i in range(num_logs):
    #     accepting_axis.plot(all_episode_steps[i], all_episode_accepting[i], color=color_violations, linewidth=0.2)
    accepting_axis.plot(plot_points, avgs, color=color_accepting)
    # accepting_axis.fill_between(plot_points, avgs - stds / 2, avgs + stds / 2, color=color_violations, alpha=0.15)
    plt.savefig(log_files[0] + "-episode_accepting_explore.png")
    plt.show()

    ####################################################
    #####                 Tests                    #####
    ####################################################

    plot_steps = plot_steps_test
    plot_points = [(i + 1) * max_steps / plot_steps for i in range(plot_steps)]

    # Scale Test für Trainingssteps
    all_test_steps = list()
    for i in range(num_logs):
        all_test_steps.append(np.array(all_episode_steps[i])[int(episodes_per_test)-1::int(episodes_per_test)])

    t_r_avg, t_r_std = scale_episodes_to_steps(num_logs=num_logs, plot_steps=plot_steps, plot_points=plot_points,
                                               steps=all_test_steps, values=all_test_rewards)

    t_a_avg, t_a_std = scale_episodes_to_steps(num_logs=num_logs, plot_steps=plot_steps, plot_points=plot_points,
                                               steps=all_test_steps, values=all_test_accepting)

    # Prepare Plot
    fig, reward_axis = plt.subplots()
    accepting_axis = reward_axis.twinx()  # instantiate a second axes that shares the same x-axis
    accepting_axis.set_xlabel('Schritte')
    accepting_axis.set_ylabel('akzeptierender Zustandsautomat', color=color_accepting)
    accepting_axis.tick_params(axis='y', labelcolor=color_accepting)
    accepting_axis.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
    accepting_axis.set_ylim([-0.02, 1.02])

    reward_axis.set_ylabel('Belohnung', color=color_reward)  # we already handled the x-label with ax1
    reward_axis.tick_params(axis='y', labelcolor=color_reward)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Plot Data
    avgs, stds, stds2 = get_avg_std_for_plot(plot_steps=plot_steps, plot_points=plot_points, averages=t_r_avg,
                                             stds=t_r_std)
    # for i in range(num_logs):
    #     reward_axis.plot(all_test_steps[i], all_test_rewards[i], color=color_reward, linewidth=0.2)
    reward_axis.plot(plot_points, avgs, color=color_reward)
    reward_axis.fill_between(plot_points, avgs - (stds+stds2) / 2, avgs + (stds+stds2) / 2, color=color_reward, alpha=0.15)
    avgs, stds, stds2 = get_avg_std_for_plot(plot_steps=plot_steps, plot_points=plot_points, averages=t_a_avg,
                                             stds=t_a_std)
    accepting_axis.plot(plot_points, avgs, color=color_accepting)
    # accepting_axis.fill_between(plot_points, avgs - stds / 2, avgs + stds / 2, color=color_violations, alpha=0.15)
    plt.savefig(log_files[0] + "-test_accepting_reward.png")
    plt.show()

    ####################################################
    #####                 Loss                     #####
    ####################################################
    if len(all_loss) == 0:
        return
    # cut loss after first loss stops
    loss_length = len(min(all_loss, key=len))
    for i in range(num_logs):
        all_loss[i] = all_loss[i][:loss_length]
    all_loss = np.array(all_loss)

    plot_points = [(i + 1) * max_steps / loss_length for i in range(loss_length)]

    losses = np.empty((loss_length, num_logs))
    for i in range(loss_length):
        losses[i] = all_loss[:, i]

    all_loss_average = np.average(losses, axis=1)
    all_loss_std = np.std(losses, axis=1)

    # plot average
    plt.plot(plot_points, all_loss_average, color=color_loss)
    # for i in range(num_logs):
    #     plt.plot(plot_points, all_loss[i], color=color_loss, linewidth=0.2)
    plt.plot(plot_points, all_loss_average, color=color_loss)
    # plot variance
    plt.fill_between(plot_points,
                     all_loss_average - all_loss_std / 2,
                     all_loss_average + all_loss_std / 2,
                     color=color_loss, alpha=0.2)
    plt.ylabel('Loss')
    plt.xlabel('Lernschritt')
    plt.savefig(log_files[0] + "-loss.png")
    plt.show()


# Scale the values to plot_steps using steps
# outputs averages and standard deviations of values for each plot_step
def scale_episodes_to_steps(num_logs=10, plot_steps=100, plot_points=range(100), steps=[], values=[]):
    averages = np.empty((num_logs, plot_steps))
    stds = np.empty((num_logs, plot_steps))

    for i in range(num_logs):
        step_avg = list()
        step_std = list()
        episode_index = 0
        for plot_point in plot_points:
            last_index = episode_index
            episode_index = np.argmax(np.array(steps[i]) >= plot_point)

            # add values from old up to new index
            scaled_values = values[i][last_index:episode_index]
            step_avg.append(np.average(scaled_values))
            step_std.append(np.std(scaled_values))
        averages[i] = step_avg
        stds[i] = step_std

    averages = np.array(averages)
    averages = np.ma.masked_array(averages, np.isnan(averages))

    stds = np.array(stds)
    stds = np.ma.masked_array(stds, np.isnan(stds))

    return averages, stds


# averages and stds with shape (num_logs, plot_steps)
# Converts to shape(plot_steps, num_logs) and plots
def get_avg_std_for_plot(plot_steps=100, plot_points=range(100), averages=[], stds=[], color='k'):
    avgs_to_plot = np.empty(plot_steps)
    stds_to_plot = np.empty(plot_steps)
    stds_to_plot2 = np.empty(plot_steps)

    # calculate averages over all log_files
    for i in range(plot_steps):
        avgs_to_plot[i] = np.average(averages[:, i])
        # variance of each runs steps
        stds_to_plot[i] = np.std(averages[:, i])
        # variance between runs
        stds_to_plot2[i] = np.average(stds[:, i])

    return avgs_to_plot, stds_to_plot, stds_to_plot2


# Plot loss and variance over learning steps
def plot_loss(log_files=[]):
    num_logs = len(log_files)

    all_loss = list()

    for i in range(len(log_files)):
        log_file = log_files[i]
        loss_file = open(log_file + "-loss.csv")
        loss = np.genfromtxt(loss_file, dtype=None, delimiter=",")

        all_loss.append(loss)

    all_loss = np.array(all_loss, dtype=object)
    length = len(max(all_loss, key=len))
    all_loss_average = np.zeros(length)
    all_loss_var = np.zeros(length)

    # manually construct average and std, because inner array length can vary...
    for i in range(length):
        values = list()
        for j in range(num_logs):
            try:
                values.append(all_loss[j][i])
            except IndexError:
                pass
        all_loss_average[i] = np.average(values)
        all_loss_var[i] = np.std(values)

    # plot average
    plt.plot(all_loss_average, color=color_loss)
    # plot variance
    plt.fill_between(range(length),
                     all_loss_average - all_loss_var /2,
                     all_loss_average + all_loss_var /2,
                     color=color_loss, alpha=0.2)
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Lernschritt')
    plt.savefig(log_files[0] + "-loss.png")
    plt.show()


# Plot average loss over learning steps. Ignore logs that run out.
def plot_loss2(log_files=[]):
    all_loss = list()

    for i in range(len(log_files)):
        log_file = log_files[i]
        loss_file = open(log_file + "-loss.csv")
        loss = np.genfromtxt(loss_file, dtype=None, delimiter=",")
        all_loss.append(loss)
        # plt.plot(loss, color=color_loss, alpha=.2)


    length = len(max(all_loss, key=len))

    for i in range(len(log_files)):
        cur_length = len(all_loss[i])
        x = np.arange(1.0, length+0.0001, length/cur_length)
        plt.plot(x, all_loss[i], color=color_loss, alpha=.4)

    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Lernschritt')
    plt.xticks([])
    plt.savefig(log_files[0] + "-loss_stretched.png")
    plt.show()


# Plot all agents losses over learning steps
def plot_loss3(log_files=[]):
    for i in range(len(log_files)):
        log_file = log_files[i]
        loss_file = open(log_file + "-loss.csv")
        loss = np.genfromtxt(loss_file, dtype=None, delimiter=",")
        plt.plot(loss, color=color_loss, alpha=.4)

    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Lernschritt')
    plt.savefig(log_files[0] + "-loss_all.png")
    plt.show()


# Plot average reward and explore_rate over episodes
def plot_episode_reward_explore(log_files=[]):
    # Reward und explore_rate
    fig, explorerate_axis = plt.subplots()
    plt.title('Belohnung in Episoden')
    explorerate_axis.set_xlabel('Episode')
    explorerate_axis.set_ylabel('Erkundungsrate', color=color_explore)
    explorerate_axis.tick_params(axis='y', labelcolor=color_explore)

    reward_axis = explorerate_axis.twinx()  # instantiate a second axes that shares the same x-axis
    reward_axis.set_ylabel('Belohnung', color=color_reward)  # we already handled the x-label with ax1
    reward_axis.tick_params(axis='y', labelcolor=color_reward)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    all_rewards = list()
    all_explore_rate = list()

    for i in range(len(log_files)):
        log_file = log_files[i]
        episode_file = open(log_file + "-episode.csv")
        episode = np.genfromtxt(episode_file, dtype=None, delimiter=",")

        rewards = list()
        explore_rate = list()

        # Extract data
        for line in episode:
            rewards.append(line[1])
            explore_rate.append(line[3])
        # plot
        explorerate_axis.plot(explore_rate, color=color_explore, alpha=.2)
        reward_axis.plot(rewards, color=color_reward, alpha=.1)

        all_rewards.append(rewards)
        all_explore_rate.append(explore_rate)

    all_rewards = np.array(all_rewards)
    all_explore_rate = np.array(all_explore_rate)

    all_rewards_average = np.zeros(len(all_rewards[0]))
    all_explore_rate_average = np.zeros(len(all_rewards[0]))

    for i in range(len(all_rewards[0])):
        all_rewards_average[i] = np.average(all_rewards[:, i])
        all_explore_rate_average[i] = np.average(all_explore_rate[:, i])

    # plot average
    explorerate_axis.plot(all_explore_rate_average, color=color_explore)
    reward_axis.plot(all_rewards_average, color=color_reward)

    plt.show()

# Plot average reward and accepting state_machine count over test-episodes
def plot_test_reward_accepting(log_files=[]):
    # Reward und explore_rate
    fig, reward_axis = plt.subplots()
    plt.title('Belohnung in Episoden')
    reward_axis.set_xlabel('Episode')
    reward_axis.set_ylabel('Belohnung', color=color_reward)
    reward_axis.tick_params(axis='y', labelcolor=color_reward)

    accepting_axis = reward_axis.twinx()
    accepting_axis.set_ylabel('Episoden mit akzeptierendem Zustandsautomaten', color=color_accepting)
    accepting_axis.tick_params(axis='y', labelcolor=color_accepting)
    fig.tight_layout()

    all_rewards = list()
    all_accepting = list()

    for i in range(len(log_files)):
        log_file = log_files[i]
        episode_file = open(log_file + "-test.csv")
        episode = np.genfromtxt(episode_file, dtype=None, delimiter=",")

        rewards = list()
        accepting = list()
        times_accepting = 0

        # Extract data
        for line in episode:
            rewards.append(line[1])
            times_accepting += line[2]
            accepting.append(times_accepting)
        # plot
        reward_axis.plot(rewards, color=color_reward, alpha=.2)
        accepting_axis.plot(accepting, color=color_accepting, alpha=.2)

        all_rewards.append(rewards)
        all_accepting.append(accepting)

    all_rewards = np.array(all_rewards)
    all_accepting = np.array(all_accepting)

    base = np.array(range(len(all_rewards[0]))) + 1
    accepting_axis.plot(base, color='k')

    all_rewards_average = np.zeros(len(all_rewards[0]))
    all_accepting_average = np.zeros(len(all_rewards[0]))

    for i in range(len(all_rewards[0])):
        all_rewards_average[i] = np.average(all_rewards[:, i])
        all_accepting_average[i] = np.average(all_accepting[:, i])

    # plot average
    reward_axis.plot(all_rewards_average, color=color_reward)
    accepting_axis.plot(all_accepting_average, color=color_accepting)

    plt.show()


# Plot average reward and explore_rate over episodes
def plot_episode_reward_explore2(log_files=[]):
    num_logs = len(log_files)

    # Reward und explore_rate
    fig, explorerate_axis = plt.subplots()
    plt.title('Belohnung in Episoden')
    explorerate_axis.set_xlabel('Episode')
    explorerate_axis.set_ylabel('Erkundungsrate', color=color_explore)
    explorerate_axis.tick_params(axis='y', labelcolor=color_explore)

    reward_axis = explorerate_axis.twinx()  # instantiate a second axes that shares the same x-axis
    reward_axis.set_ylabel('Belohnung', color=color_reward)  # we already handled the x-label with ax1
    reward_axis.tick_params(axis='y', labelcolor=color_reward)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    all_rewards = list()
    all_explore_rate = list()

    for i in range(len(log_files)):
        log_file = log_files[i]
        episode_file = open(log_file + "-episode.csv")
        episode = np.genfromtxt(episode_file, dtype=None, delimiter=",")

        rewards = list()
        explore_rate = list()

        # Extract data
        for line in episode:
            rewards.append(line[1])
            explore_rate.append(line[3])

        all_rewards.append(rewards)
        all_explore_rate.append(explore_rate)

    all_rewards = np.array(all_rewards)
    all_explore_rate = np.array(all_explore_rate)

    length = len(all_rewards[0])

    all_rewards_2 = np.zeros((length, num_logs))
    all_explore_rate_2 = np.zeros((length, num_logs))

    for i in range(length):
        all_rewards_2[i] = all_rewards[:, i]
        all_explore_rate_2[i] = all_explore_rate[:, i]

    all_rewards_average = np.average(all_rewards_2, axis=1)
    all_explore_rate_average = np.average(all_explore_rate_2, axis=1)

    all_rewards_var = np.std(all_rewards_2, axis=1)
    all_explore_rate_var = np.std(all_explore_rate_2, axis=1)

    # plot average
    explorerate_axis.plot(all_explore_rate_average, color=color_explore)
    reward_axis.plot(all_rewards_average, color=color_reward)

    # plot variance
    reward_axis.fill_between(range(length),
                             all_rewards_average - all_rewards_var/2,
                             all_rewards_average + all_rewards_var/2,
                             color=color_reward, alpha=0.2)
    explorerate_axis.fill_between(range(length),
                                  all_explore_rate_average - all_explore_rate_var/2,
                                  all_explore_rate_average + all_explore_rate_var/2,
                                  color=color_explore, alpha=0.2)

    plt.savefig(log_files[0] + "-episode_reward_explore.png")
    plt.show()


# Plot average reward and accepting state_machine count over test-episodes
def plot_test_reward_accepting2(log_files=[]):
    num_logs = len(log_files)

    # Reward und explore_rate
    fig, reward_axis = plt.subplots()
    plt.title('Testergebnisse')
    reward_axis.set_xlabel('Episode')
    reward_axis.set_ylabel('Belohnung', color=color_reward)
    reward_axis.tick_params(axis='y', labelcolor=color_reward)

    accepting_axis = reward_axis.twinx()
    accepting_axis.set_ylabel('Episoden mit akzeptierendem Zustandsautomaten', color=color_accepting)
    accepting_axis.tick_params(axis='y', labelcolor=color_accepting)
    fig.tight_layout()

    all_rewards = list()
    all_accepting = list()

    for i in range(len(log_files)):
        log_file = log_files[i]
        episode_file = open(log_file + "-test.csv")
        episode = np.genfromtxt(episode_file, dtype=None, delimiter=",")

        rewards = list()
        accepting = list()
        times_accepting = 0

        # Extract data
        for line in episode:
            rewards.append(line[1])
            times_accepting += line[2]
            accepting.append(times_accepting)

        all_rewards.append(rewards)
        all_accepting.append(accepting)

    all_rewards = np.array(all_rewards)
    all_accepting = np.array(all_accepting)

    base = np.array(range(len(all_rewards[0]))) + 1
    accepting_axis.plot(base, color='k')

    length = len(all_rewards[0])

    all_rewards_2 = np.zeros((length, num_logs))
    all_accepting_2 = np.zeros((length, num_logs))

    for i in range(length):
        all_rewards_2[i] = all_rewards[:, i]
        all_accepting_2[i] = all_accepting[:, i]

    all_rewards_average = np.average(all_rewards_2, axis=1)
    all_accepting_average = np.average(all_accepting_2, axis=1)

    all_rewards_var = np.std(all_rewards_2, axis=1)
    all_accepting_var = np.std(all_accepting_2, axis=1)

    # plot average
    accepting_axis.plot(all_accepting_average, color=color_accepting)
    reward_axis.plot(all_rewards_average, color=color_reward)

    # plot variance
    reward_axis.fill_between(range(length),
                             all_rewards_average - all_rewards_var/2,
                             all_rewards_average + all_rewards_var/2,
                             color=color_reward, alpha=0.2)
    accepting_axis.fill_between(range(length),
                                all_accepting_average - all_accepting_var / 2,
                                all_accepting_average + all_accepting_var / 2,
                                color=color_accepting, alpha=0.2)

    plt.savefig(log_files[0] + "-test_accepting_reward.png")
    plt.show()


# Plots all kinds of information for a single run of an agent
def plot(log_file="D:\\Masterarbeit\\masterarbeit\\models\\default", episode_group=5):
    ### LOSS_PLOT
    try:
        loss_file = open(log_file + "-loss.csv")
        loss = np.genfromtxt(loss_file, dtype=None, delimiter=",")

        # summarize history for loss
        plt.plot(loss, linewidth=0.3, color=color_loss)
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('Lernschritt')
        plt.savefig(log_file + "-loss.png")
        plt.show()
    except FileNotFoundError:
        print("file not found. not plotting loss.")


    ### EPISODE PLOTS
    try:
        episode_file = open(log_file + "-episode.csv")
        episode = np.genfromtxt(episode_file, dtype=None, delimiter=",")

        rewards = list()
        steps = list()
        explore_rate = list()
        accepting = list()
        reward_batch = deque(maxlen=5)
        rewards_batch = list()
        times_accepting = 0
        for line in episode:
            steps.append(line[0])
            rewards.append(line[1])
            reward_batch.append(line[1])
            rewards_batch.append(np.average(np.array(reward_batch)))
            explore_rate.append(line[3])
            times_accepting += line[2]
            accepting.append(times_accepting)

        base = np.array(range(accepting.__len__())) + 1
        reward_per_step = np.array(rewards) / np.array(steps)

        # reward normal
        plt.plot(rewards)
        plt.title('Belohnung in Episoden')
        plt.ylabel('Belohnung')
        plt.xlabel('Episode')
        plt.savefig(log_file + "-episode_reward.png")
        plt.show()

        # Reward und explore_rate
        fig, ax1 = plt.subplots()
        plt.title('Belohnung in Episoden')
        color = 'tab:orange'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Erkundungsrate', color=color)
        ax1.plot(explore_rate, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Belohnung', color=color)  # we already handled the x-label with ax1
        ax2.plot(rewards, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(log_file + "-episode_reward_explore.png")
        plt.show()

        # Reward und Reward/step
        fig, ax1 = plt.subplots()
        plt.title('Belohnung in Episoden')
        color = 'tab:cyan'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Belohnung/Schritt', color=color)
        ax1.plot(reward_per_step, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Belohnung', color=color)  # we already handled the x-label with ax1
        ax2.plot(rewards, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(log_file + "-episode_rewards.png")
        plt.show()

        # Accepting und Explore_rate
        fig, ax1 = plt.subplots()
        plt.title('Verhalten im Training')
        color = 'tab:orange'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Erkundungsrate', color=color)
        ax1.plot(explore_rate, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('Verletzungen Automat', color=color)  # we already handled the x-label with ax1
        ax2.plot(base, color='k')
        ax2.plot(accepting, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(log_file + "-episode_accepting_explore.png")
        plt.show()
    except FileNotFoundError:
        print("file not found. not plotting episode.")

    ### TEST PLOTS
    try:
        test_file = open(log_file + "-test.csv")
        test = np.genfromtxt(test_file, dtype=None, delimiter=",")

        rewards = list()
        steps = list()
        accepting = list()
        times_accepting = 0
        for line in test:
            steps.append(line[0])
            rewards.append(line[1])
            times_accepting += line[2]
            accepting.append(times_accepting)

        print("Violations in tests: ", times_accepting)
        base = np.array(range(accepting.__len__())) + 1
        reward_per_step = np.array(rewards) / np.array(steps)

        # Reward und Reward/step
        fig, ax1 = plt.subplots()
        plt.title('Belohnung in Tests')
        color = 'tab:cyan'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Belohnung/Schritt', color=color)
        ax1.plot(reward_per_step, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Belohnung', color=color)  # we already handled the x-label with ax1
        ax2.plot(rewards, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(log_file + "-test_rewards.png")
        plt.show()

        # Accepting und Reward
        fig, ax1 = plt.subplots()
        plt.title('Testergebnisse')
        color = 'tab:blue'
        ax1.set_xlabel('Testepisode')
        ax1.set_ylabel('Belohnung', color=color)
        ax1.plot(rewards, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('Episoden mit akzeptierenden Zustandsautomaten', color=color)  # we already handled the x-label with ax1
        ax2.plot(base, color='k')
        ax2.plot(accepting, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(log_file + "-test_accepting_reward.png")
        plt.show()
    except FileNotFoundError:
        print("file not found. not plotting tests.")

    ### STEP_PLOTS
    try:
        step_file = open(log_file + "-step.csv")
        test = np.genfromtxt(step_file, dtype=None, delimiter=",")

        rewards = list()
        total_reward = 0
        accepting = list()
        explore_rate = list()
        times_accepting = 0
        for line in test:
            total_reward += line[0]
            rewards.append(total_reward)
            explore_rate.append(line[2])
            times_accepting += line[1]
            accepting.append(times_accepting)

        base = np.array(range(accepting.__len__())) + 1

        # Reward
        plt.plot(rewards)
        plt.title('Belohnung in Schritten')
        plt.ylabel('Belohnung')
        plt.xlabel('Schritt')
        plt.savefig(log_file + "-step_reward.png")
        plt.show()

        # Accepting und Explore_rate
        fig, ax1 = plt.subplots()
        plt.title('Zustandsautomat in Schritten')
        color = 'tab:orange'
        ax1.set_xlabel('Schritt')
        ax1.set_ylabel('Erkundungsrate', color=color)
        ax1.plot(explore_rate, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('Verletzungen Automat', color=color)  # we already handled the x-label with ax1
        ax2.plot(base, color='k')
        ax2.plot(accepting, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(log_file + "-step_accepting_explore.png")
        plt.show()
    except FileNotFoundError:
        print("file not found. not plotting steps.")


# log_files = ['D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-0', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-1',
#              'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-2', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-3',
#              'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-4', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-5',
#              'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-6', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-7',
#              'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-8', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-9',
#              'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-10', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-11',
#              'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-12', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-13',
#              'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-14', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-15',
#              'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-16', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-17',
#              'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-18', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace2\\boatrace-qtable-rewardsub2-19']
# #
# log_files = ['D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-base-0', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-base-1', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-base-2', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-base-3', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-base-4', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-base-5', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-base-6', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-base-7', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-base-8', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-base-9']
# log_files = ['D:\\Masterarbeit\\masterarbeit\\models\\box\\box-qtable-base-0', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-qtable-base-1', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-qtable-base-2', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-qtable-base-3', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-qtable-base-4', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-qtable-base-5', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-qtable-base-6', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-qtable-base-7', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-qtable-base-8', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-qtable-base-9']
# log_files = ['D:\\Masterarbeit\\masterarbeit\\models\\box\\box-dqn-ignore2-0', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-dqn-ignore2-1', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-dqn-ignore2-2', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-dqn-ignore2-3', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-dqn-ignore2-4', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-dqn-ignore2-5', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-dqn-ignore2-6', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-dqn-ignore2-7', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-dqn-ignore2-8', 'D:\\Masterarbeit\\masterarbeit\\models\\box\\box-dqn-ignore2-9']
# log_files = ['D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-nostates-0', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-nostates-1', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-nostates-2', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-nostates-3', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-nostates-4', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-nostates-5', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-nostates-6', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-nostates-7', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-nostates-8', 'D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-nostates-9']
# plot_multiple(log_files, limit_steps=20000, plot_steps=500, plot_steps_test=125)
# plot_loss(log_files=log_files)
# plot_loss2(log_files=log_files)
# plot_loss3(log_files=log_files)
# plot_episode_reward_explore2(log_files=log_files)
# plot_test_reward_accepting2(log_files=log_files)

# plot(log_file="D:\\Masterarbeit\\masterarbeit\\models\\boatrace\\boatrace-dqn-overfit", episode_group=8)
# plot_episode_reward_explore([
#     'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-0',
#     'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-1',
#     'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-2',
#     'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-3',
#     'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-4',
#     # 'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-5',
#     # 'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-6',
#     # 'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-7',
#     # 'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-8',
#     # 'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-9'
# ])
# plot_test_reward_accepting([
#     'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-0',
#     'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-1',
#     'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-2',
#     'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-3',
#     'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-4',
#     # 'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-5',
#     # 'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-6',
#     # 'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-7',
#     # 'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-8',
#     # 'D:\\Masterarbeit\\masterarbeit\\models\\box2\\box-qtable-9'
# ])
