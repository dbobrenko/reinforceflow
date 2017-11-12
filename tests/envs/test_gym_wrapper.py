from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import numpy.testing as npt
from gym import spaces
from reinforceflow.envs.gym_wrapper import _make_gym2rf_converter
from reinforceflow.envs.gym_wrapper import _make_rf2gym_converter


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

    def test_gym2rf_converter_discrete(self):
        converter = _make_gym2rf_converter(self.space_d)
        npt.assert_equal(converter(self.gym_out_d), self.rf_out_d)

    def test_gym2rf_converter_box(self):
        converter = _make_gym2rf_converter(self.space_c)
        npt.assert_equal(converter(self.gym_out_c), self.rf_out_c)

    def test_gym2rf_converter_binary(self):
        converter = _make_gym2rf_converter(self.space_b)
        npt.assert_equal(converter(self.gym_out_b), self.rf_out_b)

    def test_gym2rf_converter_tuple(self):
        converter = _make_gym2rf_converter(self.space_t)
        _compare_recursively(converter(self.gym_out_t), self.rf_out_t)

    def test_rf2gym_converter_discrete(self):
        converter = _make_rf2gym_converter(self.space_d)
        assert converter(self.rf_out_d) == self.gym_out_d

    def test_rf2gym_converter_box(self):
        converter = _make_rf2gym_converter(self.space_c)
        npt.assert_equal(converter(self.rf_out_c), self.gym_out_c)

    def test_rf2gym_converter_binary(self):
        converter = _make_rf2gym_converter(self.space_b)
        npt.assert_equal(converter(self.rf_out_b), self.gym_out_b)

    def test_rf2gym_converter_tuple(self):
        converter = _make_rf2gym_converter(self.space_t)
        _compare_recursively(converter(self.rf_out_t), self.gym_out_t)
