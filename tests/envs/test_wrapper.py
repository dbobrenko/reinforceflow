from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from collections import deque

import numpy as np
import numpy.testing as npt
from gym import spaces

from reinforceflow.envs import ObservationStackWrap
from reinforceflow.envs import Vectorize


def _compare_recursively(sample1, sample2):
    for elem1, elem2 in zip(sample1, sample2):
        if isinstance(elem1, (list, tuple)):
            _compare_recursively(elem1, elem2)
        else:
            npt.assert_equal(elem1, elem2)


class TestConverters(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestConverters, self).__init__(*args, **kwargs)
        self.space_d = spaces.Discrete(4)
        self.gym_out_d = 2
        self.rf_out_d = [0, 0, 1, 0]

        self.space_c = spaces.Box(-1, 1, [2, 4])
        self.gym_out_c = np.random.uniform(low=-1, high=1, size=(2, 4))
        self.rf_out_c = self.gym_out_c

        self.space_b = spaces.MultiBinary(4)
        self.gym_out_b = [0, 1, 0, 1]
        self.rf_out_b = [[1, 0], [0, 1], [1, 0], [0, 1]]

        self.space_t = spaces.Tuple((self.space_d,
                                     self.space_c,
                                     self.space_b,
                                     spaces.Tuple((self.space_d, self.space_c))
                                     ))
        self.gym_out_t = tuple([self.gym_out_d, self.gym_out_c, self.gym_out_b,
                                tuple([self.gym_out_d, self.gym_out_c])])
        self.rf_out_t = tuple([self.rf_out_d, self.rf_out_c, self.rf_out_b,
                               tuple([self.rf_out_d, self.rf_out_c])])

    def test_gym2vec_converter_discrete(self):
        converter = Vectorize.make_gym2vec_converter(self.space_d)
        npt.assert_equal(converter(self.gym_out_d), self.rf_out_d)

    def test_gym2vec_converter_box(self):
        converter = Vectorize.make_gym2vec_converter(self.space_c)
        npt.assert_equal(converter(self.gym_out_c), self.rf_out_c)

    def test_gym2vec_converter_binary(self):
        converter = Vectorize.make_gym2vec_converter(self.space_b)
        npt.assert_equal(converter(self.gym_out_b), self.rf_out_b)

    def test_gym2vec_converter_tuple(self):
        converter = Vectorize.make_gym2vec_converter(self.space_t)
        _compare_recursively(converter(self.gym_out_t), self.rf_out_t)

    def test_vec2gym_converter_discrete(self):
        converter = Vectorize.make_vec2gym_converter(self.space_d)
        assert converter(self.rf_out_d) == self.gym_out_d

    def test_vec2gym_converter_box(self):
        converter = Vectorize.make_vec2gym_converter(self.space_c)
        npt.assert_equal(converter(self.rf_out_c), self.gym_out_c)

    def test_vec2gym_converter_binary(self):
        converter = Vectorize.make_vec2gym_converter(self.space_b)
        npt.assert_equal(converter(self.rf_out_b), self.gym_out_b)

    def test_vec2gym_converter_tuple(self):
        converter = Vectorize.make_vec2gym_converter(self.space_t)
        _compare_recursively(converter(self.rf_out_t), self.gym_out_t)


def test_stack_initial_observation_image_gray():
    ones = np.ones((84, 84, 1))
    stack_len = 4
    desired = np.ones((84, 84, stack_len))
    result = ObservationStackWrap.stack_observations(ones, stack_len, None)
    npt.assert_equal(result, desired)


def test_stack_observation_image_gray():
    stack_obs_test(shape=(50, 30, 1), stack_len=5, num_stacks=10)


def test_stack_observation_with_len_equals_1():
    stack_obs_test(shape=(30, 30, 1), stack_len=1, num_stacks=8)


def test_stack_observation_image_rgb():
    stack_obs_test(shape=(84, 84, 3), stack_len=4, num_stacks=12)


def test_stack_observation_exotic_shape():
    stack_obs_test(shape=(4, 4, 4, 2), stack_len=5, num_stacks=22)


def stack_obs_test(shape, stack_len, num_stacks):
    stack_axis = len(shape)-1
    desired = deque(maxlen=stack_len)
    for _ in range(stack_len):
        desired.append(np.ones(shape))
    current_stack = np.concatenate(desired, stack_axis)
    stack_len = stack_len

    for i in range(num_stacks):
        new_obs = np.ones(shape) * i
        desired.append(new_obs)
        current_stack = ObservationStackWrap.stack_observations(new_obs, stack_len, current_stack)
        npt.assert_equal(current_stack, np.concatenate(desired, stack_axis))
