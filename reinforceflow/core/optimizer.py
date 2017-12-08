from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from reinforceflow.core.decay import Decay, NullDecay
from reinforceflow.utils import tensor_utils


class Optimizer(object):
    """A Factory Method wrapper around TensorFlow Optimizers."""
    def __init__(self, learning_rate, gradient_clip, lr_decay):
        self.optimizer = None
        self._grad_clip = gradient_clip
        self._opt_class = None
        self.global_step = None
        self._kwargs = {}
        self._lr_decay = Decay.create(lr_decay, learning_rate)

    @property
    def lr(self):
        return self._kwargs['learning_rate']

    @classmethod
    def create(cls, optimizer, learning_rate=7e-4, gradient_clip=40.0, lr_decay=None):
        """Initializes optimizer.
        After initialization, consider calling 'build' method.

        Args:
            optimizer (core.Optimizer or str): Optimizer object or name. Valid names:
                'adam', 'rms', 'sgd', 'momentum', 'adadelta'.
            learning_rate (float or Tensor): Learning rate.
            gradient_clip (float): Gradient norm clipping.
            lr_decay (core.Decay or str): Learning rate decay schedule. Valid names:
                'constant', 'linear', 'poly', 'exp', 'inverse', 'natural'.

        Returns (core.Optimizer):
            Optimizer instance.
        """
        if isinstance(optimizer, Optimizer):
            return optimizer

        # Create optimizer from callable object
        if callable(optimizer):
            return CustomOptimizer(optimizer,
                                   learning_rate=learning_rate,
                                   gradient_clip=gradient_clip,
                                   lr_decay=lr_decay)

        # Create optimizer from string
        elif isinstance(optimizer, six.string_types):
            if learning_rate is None:
                raise ValueError("Learning rate must be provided in order to create an optimizer.")
            o = optimizer.lower()
            if o not in OPTIMIZER_MAP:
                raise ValueError("Unknown optimizer %s. Valid: %s." % (o, ', '.join(OPTIMIZER_MAP)))
            return OPTIMIZER_MAP[o](optimizer,
                                    learning_rate=learning_rate,
                                    gradient_clip=gradient_clip,
                                    lr_decay=lr_decay)
        else:
            raise ValueError("Unknown optimizer %s. Should be either an optimizer instance, "
                             "a name string, or a subclass of TensorFlow Optimizer."
                             % str(optimizer))

    def build(self, steps, global_step, decay_step=None):
        """Builds optimizer train graph.
        Adds 'learning_rate' scalar summary.

        Args:
            steps (int): Total amount of steps
            global_step (Tensor): Optimizer update step.
            decay_step (Tensor): Step for learning rate annealing, pass None if decay isn't used.

        Returns (Operation):
            Learning rate.
        """
        if decay_step is None and not isinstance(self._lr_decay, NullDecay):
            raise ValueError('Decay step must be defined, when lr decay is used.')

        self.global_step = global_step
        self._kwargs['learning_rate'] = self._lr_decay.build(steps, decay_step)
        self.optimizer = self._opt_class(**self._kwargs)
        return self.optimizer

    def minimize(self, loss, parameters):
        grads = self.gradients(loss, parameters)
        return self.apply_gradients(grads, parameters)

    def gradients(self, loss, parameters):
        grads = tf.gradients(loss, parameters)
        if self._grad_clip:
            grads, _ = tf.clip_by_global_norm(grads, self._grad_clip)
        return grads

    def apply_gradients(self, gradients, parameters):
        grads_params = list(zip(gradients, parameters))
        with tf.device('/cpu:0'):
            tensor_utils.add_grads_summary(grads_params)
        return self.optimizer.apply_gradients(grads_params, global_step=self.global_step)


class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 use_locking=False, name="Adam", gradient_clip=40.0, lr_decay=None):
        super(Adam, self).__init__(learning_rate, gradient_clip, lr_decay)
        self._opt_class = tf.train.AdamOptimizer
        self._kwargs['learning_rate'] = learning_rate
        self._kwargs['beta1'] = beta1
        self._kwargs['beta2'] = beta2
        self._kwargs['epsilon'] = epsilon
        self._kwargs['use_locking'] = use_locking
        self._kwargs['name'] = name


class RMSProp(Optimizer):
    def __init__(self, learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False,
                 centered=False, name="RMSProp", gradient_clip=40.0, lr_decay=None):
        super(RMSProp, self).__init__(learning_rate, gradient_clip, lr_decay)
        self._opt_class = tf.train.RMSPropOptimizer
        self._kwargs['learning_rate'] = learning_rate
        self._kwargs['decay'] = decay
        self._kwargs['momentum'] = momentum
        self._kwargs['epsilon'] = epsilon
        self._kwargs['use_locking'] = use_locking
        self._kwargs['centered'] = centered
        self._kwargs['name'] = name


class SGD(Optimizer):
    def __init__(self, learning_rate, use_locking=False, name="GradientDescent",
                 gradient_clip=40.0, lr_decay=None):
        super(SGD, self).__init__(learning_rate, gradient_clip, lr_decay)
        self._opt_class = tf.train.GradientDescentOptimizer
        self._kwargs['learning_rate'] = learning_rate
        self._kwargs['use_locking'] = use_locking
        self._kwargs['name'] = name


class Momentum(Optimizer):
    def __init__(self, learning_rate, momentum, use_locking=False, name="Momentum",
                 use_nesterov=False, gradient_clip=40.0, lr_decay=None):
        super(Momentum, self).__init__(learning_rate, gradient_clip, lr_decay)
        self._opt_class = tf.train.MomentumOptimizer
        self._kwargs['learning_rate'] = learning_rate
        self._kwargs['momentum'] = momentum
        self._kwargs['use_nesterov'] = use_nesterov
        self._kwargs['use_locking'] = use_locking
        self._kwargs['name'] = name


class AdaDelta(Optimizer):
    def __init__(self, learning_rate, rho=0.95, epsilon=1e-8, use_locking=False,
                 name="Adadelta", gradient_clip=40.0, lr_decay=None):
        super(AdaDelta, self).__init__(learning_rate, gradient_clip, lr_decay)
        self._opt_class = tf.train.AdadeltaOptimizer
        self._kwargs['learning_rate'] = learning_rate
        self._kwargs['rho'] = rho
        self._kwargs['epsilon'] = epsilon
        self._kwargs['use_locking'] = use_locking
        self._kwargs['name'] = name


class CustomOptimizer(Optimizer):
    def __init__(self, optimizer, learning_rate, gradient_clip=40.0, lr_decay=None, **kwargs):
        super(CustomOptimizer, self).__init__(learning_rate, gradient_clip, lr_decay)
        self._opt_class = optimizer
        self._kwargs = kwargs


OPTIMIZER_MAP = {
    'adam': Adam,
    'rms': RMSProp,
    'sgd': SGD,
    'momentum': Momentum,
    'adadelta': AdaDelta
}
