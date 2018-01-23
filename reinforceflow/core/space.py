from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import six
import numpy as np
from gym import spaces as _spaces
from reinforceflow import utils


@six.add_metaclass(abc.ABCMeta)
class Space(object):
    def reshape(self, n):
        raise NotImplementedError


class DiscreteOneHot(Space, _spaces.Discrete):
    """Discrete Space.
    Represents single one-hot encoded discrete value.

    Args:
        n (int): Space size.
    """
    def __init__(self, n):
        n = np.asscalar(np.asarray(n))
        super(DiscreteOneHot, self).__init__(n)

    def sample(self):
        """Returns random one-hot encoded value from current space."""
        value = super(DiscreteOneHot, self).sample()
        return utils.one_hot(self.n, value)

    def contains(self, x):
        if not isinstance(x, (tuple, list, np.generic, np.ndarray)):
            return False
        return np.shape(x) == self.shape and np.sum(x) == 1 and np.max(x) == 1

    @property
    def shape(self):
        return tuple([self.n])

    def reshape(self, n):
        self.n = np.asscalar(np.asarray(n))


class Continuous(Space, _spaces.Box):
    def __init__(self, low, high, shape=None):
        super(Continuous, self).__init__(low, high, shape)

    def reshape(self, new_shape):
        low_value = np.min(self.low)
        high_value = np.min(self.high)
        self.low = low_value + np.zeros(new_shape)
        self.high = high_value + np.zeros(new_shape)


class Tuple(Space, _spaces.Tuple):
    def __init__(self, *spaces):
        self._spaces = spaces
        self._shape = tuple([space.shape for space in spaces])

    @property
    def shape(self):
        return self._shape

    @property
    def spaces(self):
        return self._spaces

    def reshape(self, new_shape):
        raise NotImplementedError("Use reshape separately for each space in Tuple.")
