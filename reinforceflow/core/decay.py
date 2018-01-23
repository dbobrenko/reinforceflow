from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
from tensorflow import train
from tensorflow.python.framework.ops import convert_to_tensor


class Decay(object):
    """A Factory Method wrapper around TensorFlow Decay functions."""
    def __init__(self):
        self._decay = None
        self._kwargs = {}

    @classmethod
    def create(cls, decay, learning_rate):
        """Initializes learning rate decay.
        After initialization, consider calling 'build' method.

        Args:
            decay (core.Decay or str): Learning rate decay schedule. Valid names:
                'constant', 'linear', 'poly', 'exp', 'inverse', 'natural'.
            learning_rate (float or Tensor): Learning rate.

        Returns (core.Decay):
            Decay instance.

        """
        if decay is None:
            return NullDecay(learning_rate)

        if isinstance(decay, Decay):
            return decay

        # Create optimizer from alias string/function
        if isinstance(decay, six.string_types):
            decay = decay.lower()
        if decay in DECAY_MAP:
            return DECAY_MAP[decay](learning_rate=learning_rate)

        # Create optimizer from callable object
        if callable(decay):
            return CustomDecay(decay, learning_rate=learning_rate)

        raise ValueError("Unknown decay %s. Should be either a name string or "
                         "or TensorFlow function. Valid: %s."
                         % (str(decay), ', '.join(DECAY_MAP)))

    def build(self, decay_steps, step=None):
        """Builds learning rate decay operation.
        Adds 'learning_rate' scalar summary.

        Args:
            decay_steps (int): Total amount of step.
            step (Tensor): Step counter.

        Returns (Operation):
            Learning rate.
        """
        self._kwargs['decay_steps'] = decay_steps
        self._kwargs['global_step'] = step
        lr = self._decay(**self._kwargs)
        with tf.device('/cpu:0'):
            tf.summary.scalar('learning_rate', lr)
        return lr


class PolynomialDecay(Decay):
    def __init__(self, learning_rate, power=1.0, end_learning_rate=0.0001, cycle=False, name=None):
        super(PolynomialDecay, self).__init__()
        self._decay = tf.train.polynomial_decay
        self._kwargs['learning_rate'] = learning_rate
        self._kwargs['power'] = power
        self._kwargs['end_learning_rate'] = end_learning_rate
        self._kwargs['cycle'] = cycle
        self._kwargs['name'] = name


class LinearDecay(PolynomialDecay):
    def __init__(self, learning_rate, end_learning_rate=0.0001, cycle=False, name=None):
        super(LinearDecay, self).__init__(learning_rate=learning_rate,
                                          power=1.0,
                                          end_learning_rate=end_learning_rate,
                                          cycle=cycle,
                                          name=name)


class ExponentialDecay(Decay):
    def __init__(self, learning_rate, decay_rate=1.0, staircase=False, name=None):
        super(ExponentialDecay, self).__init__()
        self._decay = tf.train.exponential_decay
        self._kwargs['learning_rate'] = learning_rate
        self._kwargs['decay_rate'] = decay_rate
        self._kwargs['staircase'] = staircase
        self._kwargs['name'] = name


class InverseTimeDecay(Decay):
    def __init__(self, learning_rate, decay_rate=1.0, staircase=False, name=None):
        super(InverseTimeDecay, self).__init__()
        self._decay = tf.train.inverse_time_decay
        self._kwargs['learning_rate'] = learning_rate
        self._kwargs['decay_rate'] = decay_rate
        self._kwargs['staircase'] = staircase
        self._kwargs['name'] = name


class NaturalExpDecay(Decay):
    def __init__(self, learning_rate, decay_rate=1.0, staircase=False, name=None):
        super(NaturalExpDecay, self).__init__()
        self._decay = tf.train.natural_exp_decay
        self._kwargs['learning_rate'] = learning_rate
        self._kwargs['decay_rate'] = decay_rate
        self._kwargs['staircase'] = staircase
        self._kwargs['name'] = name


class CustomDecay(Decay):
    def __init__(self, decay, **kwargs):
        super(CustomDecay, self).__init__()
        self._decay = decay
        self._kwargs = kwargs


class NullDecay(Decay):
    def __init__(self, learning_rate):
        super(NullDecay, self).__init__()
        self._decay = lambda lr, **kw: convert_to_tensor(lr, tf.float32, name="learning_rate")
        self._kwargs['lr'] = learning_rate


DECAY_MAP = {
    'constant': NullDecay,
    'linear': LinearDecay,
    'poly': PolynomialDecay,
    'exp': ExponentialDecay,
    'inverse': InverseTimeDecay,
    'natural': NaturalExpDecay,
    train.polynomial_decay: PolynomialDecay,
    train.exponential_decay: ExponentialDecay,
    train.inverse_time_decay: InverseTimeDecay,
    train.natural_exp_decay: NaturalExpDecay
}
