from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import numpy as np
import numpy.testing as npt
from reinforceflow.utils import stack_observations


def test_stack_initial_observation_image_gray():
    ones = np.ones((84, 84, 1))
    stack_len = 4
    desired = np.ones((84, 84, stack_len))
    result = stack_observations(ones, stack_len, None)
    npt.assert_equal(result, desired)


def test_stack_observation_image_gray():
    stack_obs(shape=(50, 30, 1), stack_len=5, num_stacks=10)


def test_stack_observation_with_len_equals_1():
    stack_obs(shape=(30, 30, 1), stack_len=1, num_stacks=8)


def test_stack_observation_image_rgb():
    stack_obs(shape=(84, 84, 3), stack_len=4, num_stacks=12)


def test_stack_observation_exotic_shape():
    stack_obs(shape=(4, 4, 4, 2), stack_len=5, num_stacks=22)


def stack_obs(shape, stack_len, num_stacks):
    stack_axis = len(shape)-1
    desired = deque(maxlen=stack_len)
    for _ in range(stack_len):
        desired.append(np.ones(shape))
    current_stack = np.concatenate(desired, stack_axis)
    stack_len = stack_len

    for i in range(num_stacks):
        new_obs = np.ones(shape) * i
        desired.append(new_obs)
        current_stack = stack_observations(new_obs, stack_len, current_stack)
        npt.assert_equal(current_stack, np.concatenate(desired, stack_axis))
