from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import numpy.testing as npt
from reinforceflow.core import SumTree
# TODO: add tests for SumTree.update


def test_sumtree_sum():
    size = 100000
    tree = SumTree(size)
    for i in range(size):
        tree.append(i)
    assert tree.sum() == sum(range(size))


def test_sumtree_find_idx():
    size = 100000
    tree = SumTree(size)
    for i in range(size):
        tree.append(i)
    for i in range(size):
        idx = tree.find_sum_idx(i)
        assert 0 <= idx < size, 'Index = %s' % idx


def test_sumtree_distribution():
    priors = np.array([20000.0, 30000.0, 500.0, 49500.0, 0.0])
    tree = SumTree(len(priors))
    s = int(np.sum(priors))
    expected_priors = priors / s
    received_priors = np.zeros_like(priors)
    for p in priors:
        tree.append(p)
    for i in range(0, s):
        idx = tree.find_sum_idx(i)
        received_priors[idx] += 1
    received_priors = received_priors / s
    npt.assert_almost_equal(expected_priors, received_priors, decimal=4)
