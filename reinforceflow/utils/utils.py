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


def discount(rewards, gamma, expected_reward=0.0):
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
