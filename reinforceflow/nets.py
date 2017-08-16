from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


class DQNFactory(object):
    def make(self, input_shape, output_size, trainable=True):
        return DQNModel(input_shape, output_size, trainable)


class DuelingDQNFactory(object):
    def __init__(self, dueling_type='mean', advantage_layers=(512,), value_layers=(512,)):
        self.dueling_type = dueling_type
        self.advantage_layers = advantage_layers
        self.value_layers = value_layers

    def make(self, input_shape, output_size, trainable=True):
        return DuelingDQNModel(input_shape, output_size, dueling_type=self.dueling_type,
                               advantage_layers=self.advantage_layers,
                               value_layers=self.value_layers,
                               trainable=trainable)


class MLPFactory(object):
    def __init__(self, layer_sizes=(512, 512, 512)):
        self.layer_sizes = layer_sizes

    def make(self, input_shape, output_size, trainable=True):
        return DuelingMLPModel(input_shape, output_size, layer_sizes=self.layer_sizes,
                               trainable=trainable)


class DuelingMLPFactory(object):
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


class DQNModel(object):
    def __init__(self, input_shape, output_size, trainable=True):
        output_size = np.ravel(output_size)
        if len(output_size) != 1:
            raise ValueError('Output size must be either scalar or vector.')
        output_size = int(output_size[0])
        input_ph = tf.placeholder('float32', shape=input_shape, name='inputs')
        net, end_points = _make_dqn_body(input_ph, trainable)
        net = layers.fully_connected(net, num_outputs=512, activation_fn=tf.nn.relu,
                                     scope='fc1', trainable=trainable)
        end_points['fc1'] = net
        net = layers.fully_connected(net, num_outputs=output_size, activation_fn=None,
                                     scope='out', trainable=trainable)
        end_points['outs'] = net
        self.input_ph = input_ph
        self.inference_op = net
        self.end_points = end_points


class DuelingDQNModel(object):
    def __init__(self, input_shape, output_size, dueling_type='mean',
                 advantage_layers=(512,), value_layers=(512,), trainable=True):
        output_size = np.ravel(output_size)
        if len(output_size) != 1:
            raise ValueError('Output size must be either scalar or vector.')
        output_size = int(output_size[0])
        input_ph = tf.placeholder('float32', shape=input_shape, name='inputs')
        net, end_points = _make_dqn_body(input_ph, trainable)
        out, dueling_endpoints = _make_dueling(input_layer=net,
                                               output_size=output_size,
                                               dueling_type=dueling_type,
                                               advantage_layers=advantage_layers,
                                               value_layers=value_layers,
                                               trainable=trainable)
        end_points.update(dueling_endpoints)
        self.input_ph = input_ph
        self.inference_op = net
        self.end_points = end_points


class MLPModel(object):
    def __init__(self, input_shape, output_size, layer_sizes=(512, 512, 512),
                 output_activation=None, trainable=True):
        end_points = {}
        inputs = tf.placeholder('float32', shape=input_shape, name='inputs')
        end_points['inputs'] = inputs
        net = layers.flatten(inputs)
        for i, units in enumerate(layer_sizes):
            name = 'fc%d' % i
            net = layers.fully_connected(net, num_outputs=units, activation_fn=tf.nn.relu,
                                         trainable=trainable, scope=name)
            end_points[name] = net
        net = layers.fully_connected(net, num_outputs=output_size, activation_fn=output_activation,
                                     trainable=trainable, scope='outs')
        end_points['outs'] = net
        self.input_ph = inputs
        self.inference_op = net
        self.end_points = end_points


class DuelingMLPModel(object):    
    def __init__(self, input_shape, output_size, layer_sizes=(512, 512), dueling_type='mean',
                 advantage_layers=(256,), value_layers=(256,), trainable=True):
        end_points = {}
        inputs = tf.placeholder('float32', shape=input_shape, name='inputs')
        end_points['inputs'] = inputs
        net = layers.flatten(inputs)
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
        self.input_ph = inputs
        self.inference_op = net
        self.end_points = end_points


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
    end_points['outs'] = out
    return out, end_points
