from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from termcolor import colored

from reinforceflow import logger


def _make_row(*column_messages, **kwargs):
    """Makes a formatted string.

    Args:
        *column_messages (str): Messages.
        **kwargs:

    Returns:

    """
    color = kwargs.get("color", None)
    col_size = kwargs.get("column_size", 40)
    message = ""
    for m in column_messages:
        if m is not None:
            message += str("\t\t%-" + str(col_size) + "s ") % str(m)
    return colored(message, color=color)


def _log_rows(*rows):
    table = ""
    for r in rows:
        if r is not None:
            table += "%s\n" % r
    logger.info(table)


def flush_stats(stats, name, log_progress=True, log_rewards=True, log_performance=True,
                log_hyperparams=True, maxsteps=None, writer=None):
    name = stats.agent.name if name is None else name
    if isinstance(stats, Stats):
        stats = [stats]
    stat = stats[0]
    delta_time = time.time() - stat.last_time
    optim_per_sec = (stat.agent.optimize_counter - stat.last_optimize) / delta_time
    steps = stat.agent.step
    episodes = stat.agent.episode
    obs_per_sec = (stat.agent.step - stat.last_step) / delta_time
    reward_step = 0
    lr = 0
    episode_rewards = []
    exploration = 0
    for stat in stats:
        reward_step += stat.reward_stats.reset_step_rewards()
        if stat.reward_stats.episode > 0:
            episode_rewards.append(stat.reward_stats.reset_episode_rewards())
        # exploration += stat.agent.gamma
        # lr += stat.lr
        stat.last_time = time.time()
        stat.last_step = stat.agent.step
        stat.last_optimize = stat.agent.optimize_counter

    reward_step /= len(stats)
    reward_ep = float(np.mean(episode_rewards or 0))
    exploration /= len(stats)
    lr /= len(stats)

    percent = "(%.2f%%)" % (100 * (steps / maxsteps)) if maxsteps is not None else ""
    _log_rows(colored(name, color='green', attrs=['bold']),
              _make_row('%-20s %d %s' % ('Steps', steps, percent),
                        '%-20s %d' % ('Episodes', episodes),
                        color='blue') if log_progress else None,

              _make_row('%-20s %.4f' % ('Reward/Step', reward_step),
                        '%-20s %.2f' % ('Reward/Episode', reward_ep) if reward_ep else None,
                        color='blue') if log_rewards else None,

              _make_row('%-20s %.2f' % ('Observation/Sec', obs_per_sec),
                        '%-20s %.2f' % ('Optimization/Sec', optim_per_sec),
                        color='cyan') if log_performance else None,

              # _make_row('%-20s %.2f' % ('Exploration Rate', exploration),
              #           '%-20s %.2e' % ('Learning Rate', lr),
              #           ) if log_hyperparams else None
              )

    if writer is not None:
        # TODO
        v = tf.Summary.Value
        logs = [v(tag=name+'/TotalEpisodes', simple_value=episodes),
                v(tag=name+'/ObsPerSec', simple_value=obs_per_sec),
                v(tag=name+'/OptimizePerSec', simple_value=optim_per_sec),
                v(tag=name+'/RewardPerStep', simple_value=reward_step)]
        if reward_ep > 0:
            logs.append(v(tag=name+'/RewardPerEpisode', simple_value=reward_ep))
        writer.add_summary(tf.Summary(value=logs), global_step=steps)


class Stats(object):
    def __init__(self, agent):
        """Statistics recorder.

        Args:
            tensorboard (bool): If enabled, performs tensorboard logging.
            episodic (bool): Whether environment is episodic.
            log_performance (bool): Whether to log performance (obs/sec).
            name (str): Statistics name.
        """
        self.agent = agent
        self.last_time = time.time()
        self.reward_stats = RewardStats()
        self.action_distr = {}
        self.last_step = self.agent.step
        self.last_optimize = self.agent.optimize_counter

    def add(self, actions, rewards, terms, infos):
        """Adds statistics. Expected to be called after each `gym.Env.step`.

        Args:
            rewards (list): List of rewards after performed action.
            terms (list): List of terminal states.
            infos (list): List of info returned by environment.
        """
        # rewards = [info.get('reward_raw', reward) for reward, info in zip(rewards, infos)]
        self.reward_stats.add(rewards, terms)

    def flush(self, name=None):
        flush_stats(self, name)


class RewardStats(object):
    """Keeps agent's step and episode reward statistics."""
    def __init__(self):
        self.episode_sum = 0.0
        self.step_sum = 0.0
        self._running_ep_r = 0.0
        self.step = 0
        self.episode = 0
        self.episode_min = float('+inf')
        self.episode_max = float('-inf')

    def add(self, reward, terminal):
        """Adds reward and terminal state (end of episode).
        Args:
            reward (float, np.ndarray or list): Reward.
            terminal (bool, np.ndarray or list): Whether the episode was ended.
        """
        self.step += 1
        # TODO check for batches and single
        self.step_sum += np.sum(reward)
        self._running_ep_r += np.sum(reward)
        # Episode rewards book keeping
        if np.any(terminal):
            self.episode_sum += self._running_ep_r
            if self._running_ep_r < self.episode_min:
                self.episode_min = self._running_ep_r
            if self._running_ep_r > self.episode_max:
                self.episode_max = self._running_ep_r
            self._running_ep_r = 0
            self.episode += 1

    def add_batch(self, reward_batch, terminal_batch):
        """Adds batch with rewards and terminal states (end of episode).
        Args:
            reward_batch: List with rewards after each action.
            terminal_batch: List with booleans indicating the end of the episode after each action.
        """
        assert len(reward_batch) == len(terminal_batch)
        if not np.any(terminal_batch):
            sum_batch = np.sum(reward_batch)
            self.step += len(reward_batch)
            self.step_sum += sum_batch
            self._running_ep_r += sum_batch
            return
        # If batch contains terminal state, add by element
        for reward, term in zip(reward_batch, terminal_batch):
            self.add(reward, term)

    def step_average(self):
        """Computes average reward per step."""
        return self.step_sum / (self.step or 1)

    def episode_average(self):
        """Computes average reward per episode."""
        return self.episode_sum / (self.episode or 1)

    def reset_step_rewards(self):
        """Resets step counters.
        Returns: Average reward per step.
        """
        step = self.step_average()
        self.step_sum = 0.0
        self.step = 0
        return step

    def reset_episode_rewards(self):
        """Resets episode counters.
        Returns: Average reward per episode.
        """
        ep = self.episode_average()
        self.episode_sum = 0.0
        self.episode = 0
        self.episode_min = float('+inf')
        self.episode_max = float('-inf')
        return ep

    def reset(self):
        """Resets all counters.
        Returns: Average reward per step, Average reward per episode.
        """
        step = self.reset_step_rewards()
        ep = self.reset_episode_rewards()
        return step, ep

