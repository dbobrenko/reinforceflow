from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def stack_observations(obs, stack_len, obs_stack=None):
    """Stacks observations along last axis.

    Args:
        obs (numpy.ndarray): Observation.
        stack_len (int): Stack's total length.
        obs_stack (numpy.ndarray): Current stack of observations.
            If None, passed `obs` will be repeated for `stack_len` times.

    Returns (numpy.ndarray):
        Stack of observations.
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


class IncrementalAverage(object):
    """Incremental average counter."""
    def __init__(self):
        self._total = 0.0
        self._counter = 0
        self._min = float('+inf')
        self._max = float('-inf')

    def add(self, value):
        """Adds value."""
        self._total += value
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value
        self._counter += 1

    def add_batch(self, batch):
        """Adds batch of values."""
        self._total += np.sum(batch)
        self._counter += len(batch)
        value_min = np.min(batch)
        if value_min < self._min:
            self._min = value_min
        value_max = np.max(batch)
        if value_max > self._max:
            self._max = value_max

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    @property
    def sum(self):
        return self._total

    @property
    def length(self):
        return self._counter

    def compute_average(self):
        """Computes incremental average."""
        return self._total / (self._counter or 1)

    def reset(self):
        """Computes and resets incremental average."""
        average = self.compute_average()
        self._total = 0.0
        self._counter = 0
        self._min = float('+inf')
        self._max = float('-inf')
        return average
