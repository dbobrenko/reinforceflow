from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import time

import numpy as np
import tensorflow as tf

from reinforceflow import logger


class ThreadStats(object):
    def __init__(self, thread_stats, log_prefix='', file_writer=None,
                 initial_step=0, log_on_term=True):
        self.thread_stats = thread_stats
        self.writer = file_writer
        self.log_prefix = log_prefix
        self._scope = self.log_prefix + '/' if self.log_prefix else ''
        self.last_step = initial_step
        self.last_time = time.time()
        self.log_on_term = log_on_term

    def _reset_average(self):
        thread_step_av = []
        thread_episode_av = []
        for t in self.thread_stats:
            thread_step, thread_ep = t.reward_stats.reset()
            thread_step_av.append(thread_step)
            thread_episode_av.append(thread_ep)
        step_av = float(np.mean(thread_step_av))
        episode_av = float(np.mean(thread_episode_av))
        episode_max = float(np.max(thread_episode_av))
        return step_av, episode_av, episode_max

    def _prepare_episode_reward(self, step):
        # If some of the threads have no completed episodes yet.
        if not np.all([t.reward_stats.episode > 0 for t in self.thread_stats]):
            episode_av = 'Not Ready'
            episode_max = 'Not Ready'
        else:
            thread_step_av = [t.reward_stats.episode_average() for t in self.thread_stats]
            episode_av = float(np.mean(thread_step_av))
            episode_max = float(np.max(thread_step_av))

            if self.writer:
                value = tf.Summary.Value
                tf_logs = [value(tag=self._scope+'av_ep_R', simple_value=episode_av),
                           value(tag=self._scope+'max_ep_R', simple_value=episode_max)]
                self.writer.add_summary(tf.Summary(value=tf_logs), global_step=step)
            episode_av = "%0.2f" % episode_av
            episode_max = "%0.2f" % episode_max
            [t.reward_stats.reset_episode_rewards() for t in self.thread_stats]
        return "Episode R Average: %s (Max Thread: %s). " % (episode_av, episode_max)

    def flush(self, step, episode):
        step_av = float(np.mean([t.reward_stats.step_average() for t in self.thread_stats]))
        log = "%s " % self.log_prefix if self.log_prefix else ''
        log += self._prepare_episode_reward(step)
        log += "Av.Step R: %.4f. " % step_av
        log += "Obs: %s. Ep: %s. " % (step, episode)

        obs_per_sec = (step - self.last_step) / (time.time() - self.last_time)
        log += "Observation/sec: %0.2f." % obs_per_sec
        self.last_step = step
        if self.writer:
            value = tf.Summary.Value
            tf_logs = [value(tag=self._scope+'total_ep', simple_value=episode),
                       value(tag=self._scope+'obs_per_sec', simple_value=obs_per_sec),
                       value(tag=self._scope+'av_step_R', simple_value=step_av)]
            self.writer.add_summary(tf.Summary(value=tf_logs), global_step=step)
            self.writer.flush()

        logger.info(log)
        self.last_time = time.time()
        [t.reward_stats.reset_step_rewards() for t in self.thread_stats]


class Stats(object):
    def __init__(self, log_freq=600, file_writer=None, log_on_term=True, log_performance=True,
                 log_prefix=''):
        """Stats Logger wrapper. Used to log Average Episode Reward, Average Step Reward,
            observation/second, etc.

        Args:
            log_freq (int or None): Logging frequency, in seconds.
                To disable auto-logging, pass None.
            file_writer: Optionally, TensorFlow summary writer. To disable, pass None.
            log_on_term (bool): Whether to log only after terminal states.
            log_performance (bool): Whether to log performance (obs/sec).
            log_prefix (str): Agent's name. Used for scoping.
        """
        self.log_prefix = log_prefix
        self.log_freq = log_freq
        self.log_on_term = log_on_term
        self.log_performance = log_performance
        self.last_step = 0
        self.last_time = time.time()
        self.reward_stats = RewardStats()
        self.writer = file_writer
        self._scope = self.log_prefix + '/' if self.log_prefix else ''
        self._lock = threading.Lock()

    def add(self, reward, done, info, step, episode):
        """Adds statistics. Expected to be called after each `gym.Env.step`.
        Logs every `log_freq` second, if it's enabled.

        Args:
            reward (float): Reward after performed action.
            done (bool): Whether current step was terminal.
            info (dict): Info returned by environment.
            step (int): Current step.
            episode (int): Current episode.
        """
        with self._lock:
            reward = info.get('reward_unclip', reward)
            self.reward_stats.add(reward, done)
            if self.log_freq is None:
                return
            if time.time() - self.last_time < self.log_freq:
                return
            if self.log_on_term and not done:
                return
            self.flush(step, episode)

    def flush(self, step, episode):
        log = "%s " % self.log_prefix if self.log_prefix else ''

        episode_av = self.reward_stats.episode_average()
        episode_av = "%.2f" % episode_av if self.reward_stats.episode > 0 else "Not Ready"
        log += "Av.Episode R: %s. " % episode_av

        step_av = self.reward_stats.step_average()
        log += "Av.Step R: %.4f. " % step_av

        log += "Obs: %s. Ep: %s. " % (step, episode)
        obs_per_sec = (step - self.last_step) / (time.time() - self.last_time)

        if self.log_performance:
            log += "Observation/sec: %0.2f." % obs_per_sec

        if self.writer:
            value = tf.Summary.Value
            tf_logs = [value(tag=self._scope+'total_ep', simple_value=episode),
                       value(tag=self._scope+'obs_per_sec', simple_value=obs_per_sec),
                       value(tag=self._scope+'av_step_R', simple_value=step_av)]
            if self.reward_stats.episode > 0:
                tf_logs += [value(tag=self._scope+'av_ep_R', simple_value=float(episode_av))]
            self.writer.add_summary(tf.Summary(value=tf_logs), global_step=step)
            self.writer.flush()

        logger.info(log)
        self.last_step = step
        self.last_time = time.time()
        self.reward_stats.reset()


class RewardStats(object):
    """Keeps agent's step and episode reward statistics."""
    def __init__(self):
        self.episode_sum = 0.0
        self.step = 0
        self.episode = 0
        self.episode_min = float('+inf')
        self.episode_max = float('-inf')
        self._running_r = 0.0
        self._running_ep_r = 0.0

    def add(self, reward, terminal):
        """Adds reward and terminal state (end of episode).
        Args:
            reward (float): Reward.
            terminal (bool): Whether the episode was ended.
        """
        self.step += 1
        self._running_r += reward
        self._running_ep_r += reward
        # Episode rewards book keeping
        if terminal:
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
            self._running_r += sum_batch
            self._running_ep_r += sum_batch
            return
        # If batch contains terminal state, add by element
        for reward, term in zip(reward_batch, terminal_batch):
            self.add(reward, term)

    def step_average(self):
        """Computes average reward per step."""
        return self._running_r / (self.step or 1)

    def episode_average(self):
        """Computes average reward per episode."""
        return self.episode_sum / (self.episode or 1)

    def reset_step_rewards(self):
        """Resets step counters.
        Returns: Average reward per step.
        """
        step = self.step_average()
        self._running_r = 0.0
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

