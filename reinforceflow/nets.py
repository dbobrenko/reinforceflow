"""This module provides basic network architectures:
    Multi-layer Perceptron (MLPModel)
    Dueling Multi-layer Perceptron (DuelingMLPModel)
    Deep Q-Network model (DQNModel)
    Dueling Deep Q-Network model (DuelingDQNModel)

To implement a new model, compatible with ReinforceFlow agents, you should:
    1. Implement a Model, that inherits from `AbstractModel`.
    2. Implement a Factory for your Model, that inherits from `AbstractFactory`.

    The newly created factory can be passed to any agent you would like to use.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


@six.add_metaclass(abc.ABCMeta)
class AbstractFactory(object):
    def make(self, input_shape, output_size):
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class AbstractModel(object):
    """Abstract product model.
    To implement a new model, you should implement the `output` field.

    Args:
        input_shape: (nd.array) Input observation shape.
        output_size: (nd.array) Output action size.
    """
    @abc.abstractmethod
    def __init__(self, input_shape, output_size):
        output_size = np.squeeze(output_size).tolist()
        if isinstance(output_size, list) and len(output_size) != 1:
            raise ValueError('Output size must be either a scalar or vector.')
        self._input_ph = tf.placeholder('float32', shape=input_shape, name='inputs')

    @property
    def input_ph(self):
        """Input tensor placeholder."""
        return self._input_ph

    @property
    def output(self):
        """Output tensor operation."""
        raise NotImplementedError


class DQNFactory(AbstractFactory):
    """Factory for DQN Model.
    See `DQNModel`.
    """
    def make(self, input_shape, output_size, trainable=True):
        return DQNModel(input_shape, output_size, trainable)


class DuelingDQNFactory(AbstractFactory):
    """Factory for Dueling DQN Model.
    See `DuelingDQNModel`.
    """
    def __init__(self, dueling_type='mean', advantage_layers=(512,), value_layers=(512,)):
        self.dueling_type = dueling_type
        self.advantage_layers = advantage_layers
        self.value_layers = value_layers

    def make(self, input_shape, output_size, trainable=True):
        return DuelingDQNModel(input_shape, output_size, dueling_type=self.dueling_type,
                               advantage_layers=self.advantage_layers,
                               value_layers=self.value_layers,
                               trainable=trainable)


class MLPFactory(AbstractFactory):
    """Factory for Multilayer Perceptron."""
    def __init__(self, layer_sizes=(512, 512, 512)):
        self.layer_sizes = layer_sizes

    def make(self, input_shape, output_size, trainable=True):
        return MLPModel(input_shape, output_size, layer_sizes=self.layer_sizes,
                        trainable=trainable)


class DuelingMLPFactory(AbstractFactory):
    """Factory for Dueling Multilayer Perceptron.
    See `DuelingMLPModel`.
    """
    def __init__(self, layer_sizes=(512, 512), dueling_type='mean',
                 advantage_layers=(512,), value_layers=(512,)):
        self.dueling_type = dueling_type
        self.layer_sizes = layer_sizes
        self.advantage_layers = advantage_layers
        self.value_layers = value_layers

    def make(self, input_shape, output_size, trainable=True):
        return DuelingMLPModel(input_shape, output_size, layer_sizes=self.layer_sizes,
                               dueling_type=self.dueling_type,
                               advantage_layers=self.advantage_layers,
                               value_layers=self.value_layers,
                               trainable=trainable)


class A3CMLPFactory(AbstractFactory):
    """Factory for Multilayer Perceptron."""
    def __init__(self, layer_sizes=(512, 512, 512), policy_activation=tf.nn.softmax):
        self.layer_sizes = layer_sizes
        self.policy_activation = policy_activation

    def make(self, input_shape, output_size, trainable=True):
        return A3CMLPModel(input_shape, output_size, layer_sizes=self.layer_sizes,
                           policy_activation=self.policy_activation, trainable=trainable)


class A3CFFFactory(AbstractFactory):
    def __init__(self, policy_activation=tf.nn.softmax):
        self.policy_activation = policy_activation

    def make(self, input_shape, output_size, trainable=True):
        return A3CFFModel(input_shape, output_size,
                          policy_activation=self.policy_activation, trainable=trainable)


class MLPModel(AbstractModel):
    """Multilayer Perceptron."""
    def __init__(self, input_shape, output_size, layer_sizes=(512, 512, 512),
                 output_activation=None, trainable=True):
        super(MLPModel, self).__init__(input_shape, output_size)
        end_points = {}
        net = layers.flatten(self.input_ph)
        for i, units in enumerate(layer_sizes):
            name = 'fc%d' % i
            net = layers.fully_connected(net, num_outputs=units, activation_fn=tf.nn.relu,
                                         trainable=trainable, scope=name)
            end_points[name] = net
        end_points['out'] = layers.fully_connected(net, num_outputs=output_size,
                                                   activation_fn=output_activation,
                                                   trainable=trainable, scope='out')
        self.end_points = end_points

    @property
    def output(self):
        return self.end_points['out']


class DuelingMLPModel(AbstractModel):
    """Dueling Multilayer Perceptron.
    See "Dueling Network Architectures for Deep Reinforcement Learning", Wang et al., 2016.
    """
    def __init__(self, input_shape, output_size, layer_sizes=(512, 512), dueling_type='mean',
                 advantage_layers=(256,), value_layers=(256,), trainable=True):
        super(DuelingMLPModel, self).__init__(input_shape, output_size)
        end_points = {}
        net = layers.flatten(self.input_ph)
        for i, units in enumerate(layer_sizes):
            name = 'fc%d' % i
            net = layers.fully_connected(net, num_outputs=units, activation_fn=tf.nn.relu,
                                         trainable=trainable, scope=name)
            end_points[name] = net
        net, dueling_endpoints = _make_dueling(input_layer=net,
                                               output_size=output_size,
                                               dueling_type=dueling_type,
                                               advantage_layers=advantage_layers,
                                               value_layers=value_layers,
                                               trainable=trainable)
        end_points.update(dueling_endpoints)
        self._output = net
        self.end_points = end_points

    @property
    def output(self):
        return self._output


class DQNModel(AbstractModel):
    """Deep Q-Network model.
    See "Human-level control through deep reinforcement learning", Mnih et al., 2015.
    """
    def __init__(self, input_shape, output_size, trainable=True):
        super(DQNModel, self).__init__(input_shape, output_size)
        net, end_points = _make_dqn_body(self.input_ph, trainable)
        net = layers.fully_connected(net, num_outputs=512, activation_fn=tf.nn.relu,
                                     scope='fc1', trainable=trainable)
        end_points['fc1'] = net
        end_points['out'] = layers.fully_connected(net, num_outputs=output_size,
                                                   activation_fn=None, scope='out',
                                                   trainable=trainable)
        self.end_points = end_points

    @property
    def output(self):
        return self.end_points['out']


class A3CFFModel(AbstractModel):
    """Asynchronous Advantage Actor-Critic Feed-Forward model.
    See "Human-level control through deep reinforcement learning", Mnih et al., 2015.
    """
    def __init__(self, input_shape, output_size, trainable=True, policy_activation=tf.nn.softmax):
        super(A3CFFModel, self).__init__(input_shape, output_size)
        net, end_points = _make_dqn_body(self.input_ph, trainable)
        end_points['fc1'] = layers.fully_connected(net, num_outputs=512, activation_fn=tf.nn.relu,
                                                   scope='fc1', trainable=trainable)
        end_points['out_value'] = layers.fully_connected(end_points['fc1'], num_outputs=1,
                                                         activation_fn=None, scope='out_value',
                                                         trainable=trainable)
        end_points['out_value'] = tf.squeeze(end_points['out_value'])
        end_points['out_policy'] = layers.fully_connected(end_points['fc1'],
                                                          num_outputs=output_size,
                                                          activation_fn=policy_activation,
                                                          scope='out_policy', trainable=trainable)
        self.end_points = end_points
        self.output_policy = self.output

    @property
    def output(self):
        return self.end_points['out_policy']

    @property
    def output_value(self):
        return self.end_points['out_value']


class A3CMLPModel(AbstractModel):
    """Asynchronous Advantage Actor-Critic MLP model."""
    def __init__(self, input_shape, output_size, layer_sizes=(512, 512, 512),
                 policy_activation=tf.nn.softmax, trainable=True):
        super(A3CMLPModel, self).__init__(input_shape, output_size)
        end_points = {}
        net = layers.flatten(self.input_ph)
        for i, units in enumerate(layer_sizes):
            name = 'fc%d' % i
            net = layers.fully_connected(net, num_outputs=units, activation_fn=tf.nn.relu,
                                         trainable=trainable, scope=name)
            end_points[name] = net
        end_points['out_policy'] = layers.fully_connected(net, num_outputs=output_size,
                                                          activation_fn=policy_activation,
                                                          trainable=trainable, scope='out_policy')
        # end_points['out_policy'] = tf.squeeze(end_points['out_policy'])
        end_points['out_value'] = layers.fully_connected(net, num_outputs=1,
                                                         activation_fn=None, scope='out_value',
                                                         trainable=trainable)
        end_points['out_value'] = tf.squeeze(end_points['out_value'])
        self.end_points = end_points
        self.output_policy = self.output

    @property
    def output(self):
        return self.end_points['out_policy']

    @property
    def output_value(self):
        return self.end_points['out_value']


class DuelingDQNModel(AbstractModel):
    """Dueling Deep Q-Network model.
    See "Dueling Network Architectures for Deep Reinforcement Learning", Schaul et al., 2016.
    """
    def __init__(self, input_shape, output_size, dueling_type='mean',
                 advantage_layers=(512,), value_layers=(512,), trainable=True):
        super(DuelingDQNModel, self).__init__(input_shape, output_size)
        net, end_points = _make_dqn_body(self.input_ph, trainable)
        out, dueling_endpoints = _make_dueling(input_layer=net,
                                               output_size=output_size,
                                               dueling_type=dueling_type,
                                               advantage_layers=advantage_layers,
                                               value_layers=value_layers,
                                               trainable=trainable)
        end_points.update(dueling_endpoints)
        self._output = net
        self.end_points = end_points

    @property
    def output(self):
        return self._output


def _make_dqn_body(input_layer, trainable=True):
    end_points = {}
    net = layers.conv2d(inputs=input_layer,
                        num_outputs=32,
                        kernel_size=[8, 8],
                        stride=[4, 4],
                        activation_fn=tf.nn.relu,
                        padding="same",
                        scope="conv1",
                        trainable=trainable)
    end_points['conv1'] = net
    net = layers.conv2d(inputs=net,
                        num_outputs=64,
                        kernel_size=[4, 4],
                        stride=[2, 2],
                        activation_fn=tf.nn.relu,
                        padding="same",
                        scope="conv2",
                        trainable=trainable)
    end_points['conv2'] = net
    net = layers.conv2d(inputs=net,
                        num_outputs=64,
                        kernel_size=[3, 3],
                        stride=[1, 1],
                        activation_fn=tf.nn.relu,
                        padding="same",
                        scope="conv3",
                        trainable=trainable)
    end_points['conv3'] = net
    out = layers.flatten(net)
    end_points['conv3_flatten'] = out
    return out, end_points


def _make_dueling(input_layer, output_size, dueling_type='mean',
                  advantage_layers=(512,), value_layers=(512,), trainable=True):
    end_points = {}
    adv_layer = input_layer
    for i, units in enumerate(advantage_layers):
        name = 'advantage%d' % i
        adv_layer = layers.fully_connected(adv_layer, num_outputs=units,
                                           activation_fn=tf.nn.relu,
                                           trainable=trainable, scope=name)
        end_points[name] = adv_layer
    adv_layer = layers.fully_connected(adv_layer, num_outputs=output_size, activation_fn=None,
                                       scope='adv_out', trainable=trainable)
    end_points['adv_out'] = adv_layer

    value_layer = input_layer
    for i, units in enumerate(value_layers):
        name = 'value%d' % i
        value_layer = layers.fully_connected(value_layer, num_outputs=units,
                                             activation_fn=tf.nn.relu,
                                             trainable=trainable, scope=name)
        end_points[name] = value_layer
    value_layer = layers.fully_connected(value_layer, num_outputs=1, activation_fn=None,
                                         scope='value_out', trainable=trainable)
    end_points['value_out'] = value_layer
    if dueling_type == 'naive':
        out = value_layer + adv_layer
    elif dueling_type == 'mean':
        out = value_layer + (adv_layer - tf.reduce_mean(adv_layer, 1, keep_dims=True))
    elif dueling_type == 'max':
        out = value_layer + (adv_layer - tf.reduce_max(adv_layer, 1, keep_dims=True))
    else:
        raise ValueError("Unknown dueling type '%s'. Available: 'naive', 'mean', 'max'."
                         % dueling_type)
    end_points['out'] = out
    return out, end_points
