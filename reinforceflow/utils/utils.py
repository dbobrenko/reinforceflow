from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def stack_observations(obs, stack_len, obs_stack=None):
    """Stacks observations along last axis.
       New observations are appended to the existing stack,
       so the chronological order of observations looks like:
       [Obs{N}, Obs{N-1}, ..., Obs{2}, Obs{1: most recent}]

    Args:
        obs (numpy.ndarray): Observation.
        stack_len (int): Stack's total length.
        obs_stack (numpy.ndarray): Current stack of observations.
            If None, passed `obs` will be repeated for `stack_len` times.

    Returns (numpy.ndarray):
        Stacked observations along last axis.
    """
    stack_axis = np.ndim(obs) - 1
    obs_axis_len = np.shape(obs)[stack_axis]
    if obs_stack is None:
        obs_stack = obs
        # np.repeat won't work correctly, since it repeats each element separately,
        # instead of repeating each observation.
        for i in range(stack_len - 1):
            obs_stack = np.append(obs_stack, obs, axis=stack_axis)
    else:
        # Delete the oldest observation.
        # Note, that a single observation may have several depth channels e.g RGB,
        # so that we need to delete each of its channels separately.
        del_indexes = list(range(0, obs_axis_len))
        obs_previous = np.delete(obs_stack, del_indexes, axis=stack_axis)
        obs_stack = np.append(obs_previous, obs, axis=stack_axis)
    assert obs_stack.shape[stack_axis] // obs_axis_len == stack_len
    return obs_stack


def image_preprocess(obs, resize_width, resize_height, to_gray):
    """Applies basic preprocessing for image observations.

    Args:
        obs (numpy.ndarray): 2-D or 3-D uint8 type image.
        resize_width (int): Resize width. To disable resize, pass None.
        resize_height (int): Resize height. To disable resize, pass None.
        to_gray (bool): Converts image to grayscale.

    Returns (numpy.ndarray):
        Processed 3-D float type image.
    """
    processed_obs = np.squeeze(obs)
    if to_gray:
        processed_obs = cv2.cvtColor(processed_obs, cv2.COLOR_RGB2GRAY)
    if resize_height and resize_width:
        processed_obs = cv2.resize(processed_obs, (resize_height, resize_width))
    if np.ndim(processed_obs) == 2:
        processed_obs = np.expand_dims(processed_obs, 2)
    return processed_obs


def discount_rewards(rewards, gamma, expected_reward=0.0):
    """Applies reward discounting.

    Args:
        rewards (list): Rewards.
        gamma (float): Discount factor.
        expected_reward (float): Expected future reward.

    Returns (list):
        Discounted rewards
    """
    discount_sum = expected_reward
    result = [0] * len(rewards)
    for i in reversed(range(len(rewards))):
        discount_sum = rewards[i] + gamma * discount_sum
        result[i] = discount_sum
    return result


def one_hot(shape, idx):
    """Applies one-hot encoding.

    Args:
        shape (int): Shape of the output vector.
        idx (int): One-hot index.

    Returns (numpy.ndarray):
        One-hot encoded vector.
    """
    vec = np.zeros(shape, dtype=np.uint8)
    vec[idx] = 1
    return vec


class RewardStats(object):
    """Keeps agent's step and episode reward statistics."""
    def __init__(self):
        self._step_r = 0.0
        self.episode_sum = 0.0
        self.step = 0
        self.episode = 0
        self.episode_min = float('+inf')
        self.episode_max = float('-inf')
        self._ep_running_r = 0.0

    def add(self, reward, terminal):
        """Adds reward and terminal state (end of episode).
        Args:
            reward (float): Reward.
            terminal (bool): Whether the episode was ended.
        """
        self._step_r += reward
        self.step += 1
        self._ep_running_r += reward
        # Episode rewards book keeping
        if terminal:
            self.episode_sum += self._ep_running_r
            if self._ep_running_r < self.episode_min:
                self.episode_min = self._ep_running_r
            if self._ep_running_r > self.episode_max:
                self.episode_max = self._ep_running_r
            self._ep_running_r = 0
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
            self._step_r += sum_batch
            self.step += len(reward_batch)
            self._ep_running_r += sum_batch
            return
        # If batch contains terminal state, add by elements
        for reward, term in zip(reward_batch, terminal_batch):
            self.add(reward, term)

    def step_average(self):
        """Computes average rewards per step."""
        return self._step_r / (self.step or 1)

    def episode_average(self):
        """Computes average rewards per episode."""
        return self.episode_sum / (self.episode or 1)

    def reset(self):
        """Resets all counters.
        Returns: Average reward per step, Average reward per episode.
        """
        ep = self.episode_average()
        step = self.step_average()
        self._step_r = 0.0
        self.step = 0
        self.episode_sum = 0.0
        self._ep_running_r = 0.0
        self.episode = 0
        self.episode_min = float('+inf')
        self.episode_max = float('-inf')
        return step, ep
