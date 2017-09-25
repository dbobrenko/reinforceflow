from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import numpy.testing as npt
from reinforceflow.core import SumTree, MinTree


def test_sumtree_sum():
    capacity = 100000
    dataset = list(range(capacity))
    dataset_actual = list(range(capacity, 2*capacity))
    tree = SumTree(capacity)
    for i in dataset:
        tree.append(i)
    for i in dataset_actual:
        tree.append(i)
    assert tree.sum() == sum(dataset_actual)


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


def test_mintree_min():
    capacity = 100000
    dataset = list(range(capacity))
    dataset_actual = list(range(capacity, 2*capacity))
    tree = MinTree(capacity)
    for i in dataset:
        tree.append(i)
    for i in dataset_actual:
        tree.append(i)
    assert tree.min() == min(dataset_actual)
