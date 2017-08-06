from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from reinforceflow import logger

_OPTIMIZER_MAP = {
    'rms': tf.train.RMSPropOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'rmspropoptimizer': tf.train.RMSPropOptimizer,
    'adam': tf.train.AdamOptimizer,
    'adamoptimizer': tf.train.AdamOptimizer,
    'sgd': tf.train.GradientDescentOptimizer,
    'gradientdescent': tf.train.GradientDescentOptimizer,
    'gradientdescentoptimizer': tf.train.GradientDescentOptimizer
}


_DECAY_MAP = {
    'polynomial_decay': tf.train.polynomial_decay,
    'polynomial': tf.train.polynomial_decay,
    'poly': tf.train.polynomial_decay,
    'exponential_decay': tf.train.exponential_decay,
    'exponential': tf.train.exponential_decay,
    'exp': tf.train.exponential_decay
}


def create_optimizer(opt, learning_rate, optimizer_args=None, decay=None,
                     decay_args=None, global_step=None):
    if optimizer_args is None:
        optimizer_args = {}
    # Process learning rate
    if isinstance(learning_rate, ops.Tensor) and learning_rate.get_shape().ndims == 0:
        if decay:
            logger.warn("Passed learning rate is already of type Tensor. "
                        "Leaving optimizer original learning rate Tensor (%s) unchanged."
                        % learning_rate)
    elif isinstance(learning_rate, (float, int)):
        if learning_rate < 0.0:
            raise ValueError("Learning rate must be >= 0. Got: %s.", learning_rate)
        learning_rate = ops.convert_to_tensor(learning_rate, dtype=tf.float32,
                                              name="learning_rate")
        if decay:
            if global_step is None:
                raise ValueError('Global step must be specified, '
                                 'in order to use learning rate decay.')
            if decay_args is None:
                decay_args = {}
            learning_rate = create_decay(decay, learning_rate, global_step, **decay_args)
    else:
        raise ValueError("Learning rate should be 0d Tensor or float. "
                         "Got %s of type %s" % (str(learning_rate), str(type(learning_rate))))

    # Create optimizer
    if callable(opt):
        return opt(learning_rate=learning_rate, **optimizer_args), learning_rate
    elif isinstance(opt, six.string_types):
        opt = opt.lower()
        if opt not in _OPTIMIZER_MAP:
            raise ValueError("Unknown optimizer name %s. Available: %s."
                             % (opt, ', '.join(_OPTIMIZER_MAP)))
        return _OPTIMIZER_MAP[opt](learning_rate=learning_rate, **optimizer_args), learning_rate
    else:
        raise ValueError("Unknown optimizer %s. Should be either a class name string,"
                         "subclass of Optimizer or any callable object"
                         "with `learning_rate` argument." % str(opt))


def create_decay(decay, learning_rate, global_step, **kwargs):
    if callable(decay) and hasattr(decay, __name__):
        decay = decay.__name__

    if isinstance(decay, six.string_types):
        decay = decay.lower()
        if decay in ['polynomial_decay', 'polynomial', 'poly']:
            if 'decay_steps' not in kwargs:
                raise ValueError('You should specify decay_steps argument for the `%s`'
                                 ' decay function.' % decay)
            learning_rate = tf.train.polynomial_decay(learning_rate, global_step, **kwargs)
        elif decay in ['exponential_decay', 'exponential', 'exp']:
            learning_rate = tf.train.exponential_decay(learning_rate, global_step, **kwargs)
        else:
            raise ValueError('Unknown decay function %s. Available: %s'
                             % (decay, ', '.join(_DECAY_MAP)))
    else:
        raise ValueError('Decay should be either a decay function or a function name string.')
    return learning_rate


def discount_rewards(rewards, gamma, expected_reward=0.0):
    discount_sum = expected_reward
    result = [0] * len(rewards)
    for i in reversed(range(len(rewards))):
        discount_sum = rewards[i] + gamma * discount_sum
        result[i] = discount_sum
    return result


class IncrementalAverage(object):
    def __init__(self):
        self._total = 0.0
        self._counter = 0

    def add(self, value):
        self._total += value
        self._counter += 1

    def add_batch(self, batch):
        self._total += np.sum(batch)
        self._counter += len(batch)

    def compute_average(self):
        return self._total / (self._counter or 1)

    def reset(self):
        average = self.compute_average()
        self._total = 0.0
        self._counter = 0
        return average

    def __float__(self):
        return self.compute_average()

    def __repr__(self):
        return str(self.compute_average())
