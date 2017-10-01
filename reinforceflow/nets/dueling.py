from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from reinforceflow.nets import AbstractFactory, AbstractModel, make_dqn_body
from reinforceflow.core import Tuple


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

    def make(self, input_space, output_space, trainable=True):
        return DuelingMLPModel(input_space, output_space, layer_sizes=self.layer_sizes,
                               dueling_type=self.dueling_type,
                               advantage_layers=self.advantage_layers,
                               value_layers=self.value_layers,
                               trainable=trainable)


class DuelingDQNFactory(AbstractFactory):
    """Factory for Dueling DQN Model.
    See `DuelingDQNModel`.
    """
    def __init__(self, dueling_type='mean', advantage_layers=(512,), value_layers=(512,)):
        self.dueling_type = dueling_type
        self.advantage_layers = advantage_layers
        self.value_layers = value_layers

    def make(self, input_space, output_space, trainable=True):
        return DuelingDQNModel(input_space, output_space, dueling_type=self.dueling_type,
                               advantage_layers=self.advantage_layers,
                               value_layers=self.value_layers,
                               trainable=trainable)


class DuelingMLPModel(AbstractModel):
    """Dueling Multilayer Perceptron.
    See "Dueling Network Architectures for Deep Reinforcement Learning", Wang et al., 2016.
    """
    def __init__(self, input_space, output_space, layer_sizes=(512, 512), dueling_type='mean',
                 advantage_layers=(256,), value_layers=(256,), trainable=True):
        if isinstance(input_space, Tuple) or isinstance(output_space, Tuple):
            raise ValueError('For tuple action and observation spaces '
                             'consider implementing custom network architecture.')
        self._input_ph = tf.placeholder('float32', shape=[None] + list(input_space.shape),
                                        name='inputs')

        end_points = {}
        net = layers.flatten(self.input_ph)
        for i, units in enumerate(layer_sizes):
            name = 'fc%d' % i
            net = layers.fully_connected(net, num_outputs=units, activation_fn=tf.nn.relu,
                                         trainable=trainable, scope=name)
            end_points[name] = net
        net, dueling_endpoints = make_dueling_header(input_layer=net,
                                                     output_size=output_space.shape[0],
                                                     dueling_type=dueling_type,
                                                     advantage_layers=advantage_layers,
                                                     value_layers=value_layers,
                                                     trainable=trainable)
        end_points.update(dueling_endpoints)
        self._output = net
        self.end_points = end_points

    @property
    def input_ph(self):
        return self._input_ph

    @property
    def output(self):
        return self._output


class DuelingDQNModel(AbstractModel):
    """Dueling Deep Q-Network model.
    See "Dueling Network Architectures for Deep Reinforcement Learning", Schaul et al., 2016.
    """
    def __init__(self, input_space, output_space, dueling_type='mean',
                 advantage_layers=(512,), value_layers=(512,), trainable=True):
        if isinstance(input_space, Tuple) or isinstance(output_space, Tuple):
            raise ValueError('For tuple action and observation spaces '
                             'consider implementing custom network architecture.')
        self._input_ph = tf.placeholder('float32', shape=[None] + list(input_space.shape),
                                        name='inputs')
        net, end_points = make_dqn_body(self.input_ph, trainable)
        out, dueling_endpoints = make_dueling_header(input_layer=net,
                                                     output_size=output_space.shape[0],
                                                     dueling_type=dueling_type,
                                                     advantage_layers=advantage_layers,
                                                     value_layers=value_layers,
                                                     trainable=trainable)
        end_points.update(dueling_endpoints)
        self._output = net
        self.end_points = end_points

    @property
    def input_ph(self):
        return self._input_ph

    @property
    def output(self):
        return self._output


def make_dueling_header(input_layer, output_size, dueling_type='mean',
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
