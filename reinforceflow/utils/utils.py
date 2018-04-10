from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


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


def isarray(x):
    return isinstance(x, (list, tuple, set, np.ndarray))


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
    return np.asarray(result)


def discount_trajectory(rewards, terms, traj_ends, gamma, expected_values):
    """Applies reward discounting for trajectories.

    Args:
        rewards (list): Reward for each transition.
        terms (list): List of of bools indicating terminal states for each transition.
        traj_ends (list): List of bools indicating the end of trajectory for each transition.
        gamma (float): Discount factor.
        expected_values (list): Expected future reward for each transition.

    Returns (list):
        Discounted rewards.
    """
    terms = np.asarray(terms, 'bool')
    traj_ends = np.asarray(traj_ends, 'bool')
    expected_values = np.asarray(expected_values)
    bootstrap_idx = traj_ends == (terms == 0)
    zeros = np.zeros_like(traj_ends, 'float32')

    # If None, fill EV with zeros
    if expected_values is None:
        pass

    elif sum(bootstrap_idx) == len(expected_values):
        zeros[bootstrap_idx] = expected_values

    else:
        zeros[bootstrap_idx] = expected_values[bootstrap_idx]

    expected_values = zeros
    assert len(rewards) == len(terms) == len(traj_ends)
    assert sum(traj_ends) >= sum(terms)

    result = [0] * len(rewards)
    discount_sum = 0.
    for i in reversed(range(len(rewards))):
        if traj_ends[i]:
            discount_sum = expected_values[i]
        discount_sum = rewards[i] + gamma * discount_sum
        result[i] = discount_sum
    return np.asarray(result, dtype='float32')


def onehot(idx, shape):
    """Applies one-hot encoding.

    Args:
        idx (int): One-hot index.
        shape (int): Shape of the output vector.

    Returns (numpy.ndarray):
        One-hot encoded vector.
    """
    vec = np.zeros(shape, dtype=np.uint8)
    vec[idx] = 1
    return vec
