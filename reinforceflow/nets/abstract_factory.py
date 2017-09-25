"""To implement a new model, compatible with ReinforceFlow agents, you should:
    1. Implement a Model, that inherits from `AbstractModel`.
    2. Implement a Factory for your Model, that inherits from `AbstractFactory`.

    The newly created factory can be passed to any agent you would like to use.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class AbstractFactory(object):
    def make(self, input_space, output_space):
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class AbstractModel(object):
    """Abstract product model.
    To implement a new model, the following fields should be implemented:
        input_ph: Input observation placeholder.
        output: Output operation (inference(obs) -> action).
    """

    @property
    def input_ph(self):
        """Input tensor placeholder."""
        raise NotImplementedError

    @property
    def output(self):
        """Output tensor operation."""
        raise NotImplementedError
