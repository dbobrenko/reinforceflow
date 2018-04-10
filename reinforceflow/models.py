from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier_init
from reinforceflow.core import Continuous


class Model(object):
    def build_from_inputs(self, inputs, output_space, trainable=True):
        raise NotImplementedError

    def build(self, input_space, output_space, trainable=True):
        """Complies model.
        Args:
            input_space: (core.spaces.space) Observation space.
            output_space (core.spaces.space) Action space.
            trainable (bool): Whether the model is trainable.

        Returns (dict):
            Dict with model endpoints: "in", "out", etc.
        """
        input_ph = tf.placeholder('float32', shape=[None] + list(input_space.shape), name='inputs')
        return self.build_from_inputs(input_ph, output_space, trainable)


class FullyConnected(Model):
    """Fully Connected Model."""

    def __init__(self, layer_sizes=(512, 512), output_activation=None):
        self.layer_sizes = layer_sizes
        self.output_activation = output_activation

    def build_from_inputs(self, inputs, output_space, trainable=True):
        net = inputs
        for i, units in enumerate(self.layer_sizes):
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu,
                                  trainable=trainable, name='fc%d' % i)
        out = tf.layers.dense(net, units=output_space.shape[0],
                              activation=self.output_activation,
                              trainable=trainable, name='out')
        return {'input': inputs, 'value': out}


class DeepQModel(Model):
    """Deep Q-Network model as defined in:
    Human-level control through deep reinforcement learning, Mnih et al., 2015.
    Args:
        nature_arch (bool): If enabled, uses architecture as defined in Mnih et al., 2015,
            otherwise in Mnih et al., 2013.
        dueling (bool): If enabled, uses dueling head as defined in Wang et al., 2015.
    """

    def __init__(self, nature_arch=True, dueling=True):
        self.nature_arch = nature_arch
        self.dueling = dueling

    def build_from_inputs(self, inputs, output_space, trainable=True):
        net = tf.layers.conv2d(inputs=inputs,
                               filters=32 if self.nature_arch else 16,
                               kernel_size=[8, 8],
                               strides=[4, 4],
                               activation=tf.nn.relu,
                               padding="same",
                               name="conv1",
                               trainable=trainable)
        net = tf.layers.conv2d(inputs=net,
                               filters=64 if self.nature_arch else 32,
                               kernel_size=[4, 4],
                               strides=[2, 2],
                               activation=tf.nn.relu,
                               padding="same",
                               name="conv2",
                               trainable=trainable)
        if self.nature_arch:
            net = tf.layers.conv2d(inputs=net,
                                   filters=64,
                                   kernel_size=[3, 3],
                                   strides=[1, 1],
                                   activation=tf.nn.relu,
                                   padding="same",
                                   name="conv3",
                                   trainable=trainable)
        net = tf.layers.flatten(net)

        if self.dueling:
            return {'input': inputs, 'value': dueling_header(net, output_space)}

        net = tf.layers.dense(net, units=512 if self.nature_arch else 256,
                              activation=tf.nn.relu, name='fc1', trainable=trainable)
        out = tf.layers.dense(net,
                              units=output_space.shape[0], activation=None,
                              name='out', trainable=trainable)
        return {'input': inputs, 'value': out}


class ActorCriticConv(Model):
    """Actor-Critic Convolutional model."""
    def build_from_inputs(self, inputs, output_space, trainable=True):
        net = tf.layers.conv2d(inputs=inputs,
                               filters=16,
                               kernel_size=[8, 8],
                               strides=[4, 4],
                               activation=tf.nn.relu,
                               padding="same",
                               name="conv1",
                               trainable=trainable,
                               kernel_initializer=xavier_init(),
                               bias_initializer=xavier_init()
                               )
        net = tf.layers.conv2d(inputs=net,
                               filters=32,
                               kernel_size=[4, 4],
                               strides=[2, 2],
                               activation=tf.nn.relu,
                               padding="same",
                               name="conv2",
                               trainable=trainable,
                               kernel_initializer=xavier_init(),
                               bias_initializer=xavier_init()
                               )
        net = tf.layers.flatten(net)
        fc1 = tf.layers.dense(net,
                              units=256,
                              activation=tf.nn.relu,
                              name='fc1',
                              trainable=trainable,
                              kernel_initializer=xavier_init(),
                              bias_initializer=xavier_init()
                              )
        v = tf.layers.dense(fc1, units=1,
                            activation=None,
                            kernel_initializer=xavier_init(),
                            bias_initializer=xavier_init(),
                            name='out_value',
                            trainable=trainable)
        v = tf.squeeze(v, axis=1)
        out_pi = policy_head(net, output_space, trainable)
        return {'input': inputs, 'out': out_pi, 'policy': out_pi, 'value': v}


class ActorCriticFC(Model):
    """Actor-Critic Fully Connected model."""

    def __init__(self, fc_layers=(512, 512, 512)):
        self.layer_sizes = fc_layers

    def build_from_inputs(self, inputs, output_space, trainable=True):
        net = inputs
        for i, units in enumerate(self.layer_sizes):
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu,
                                  trainable=trainable, name='fc%d' % i)
        v = tf.layers.dense(net, units=1,
                            activation=None,
                            kernel_initializer=xavier_init(),
                            bias_initializer=xavier_init(),
                            name='out_value',
                            trainable=trainable)
        v = tf.squeeze(v, axis=1)
        out_pi = policy_head(net, output_space, trainable)
        return {'input': inputs, 'out': out_pi, 'policy': out_pi, 'value': v}


def policy_head(input_layer, output_space, trainable=True):
    if not isinstance(output_space, Continuous):
        pi = tf.layers.dense(input_layer, units=output_space.shape[0],
                             activation=tf.nn.softmax,
                             kernel_initializer=xavier_init(),
                             bias_initializer=xavier_init(),
                             name='out_policy',
                             trainable=trainable)
        return pi

    mean = tf.layers.dense(input_layer, units=output_space.shape[0],
                           activation=tf.nn.tanh,
                           kernel_initializer=xavier_init(),
                           bias_initializer=xavier_init(),
                           name='mean',
                           trainable=trainable)
    std = tf.layers.dense(input_layer, units=output_space.shape[0],
                          activation=tf.nn.softplus,
                          kernel_initializer=xavier_init(),
                          bias_initializer=xavier_init(),
                          name='std',
                          trainable=trainable)
    mean = tf.squeeze(mean)
    std = tf.squeeze(std)
    dist = tf.contrib.distributions.Normal(mean, std)
    pi = tf.clip_by_value(dist.sample(1),
                          output_space.low, output_space.high)
    return pi


def dueling_header(input_layer, output_space, dueling_type='mean',
                   advantage_layers=(512,), value_layers=(512,), trainable=True):
    adv_layer = input_layer
    for i, units in enumerate(advantage_layers):
        adv_layer = tf.layers.dense(adv_layer, units=units,
                                    activation=tf.nn.relu, trainable=trainable,
                                    name='advantage%d' % i)
    adv_layer = tf.layers.dense(adv_layer, units=output_space.shape[0],
                                activation=None, name='adv_out', trainable=trainable)

    value_layer = input_layer
    for i, units in enumerate(value_layers):
        value_layer = tf.layers.dense(value_layer, units=units,
                                      activation=tf.nn.relu, trainable=trainable,
                                      name='value%d' % i)
    value_layer = tf.layers.dense(value_layer, units=1, activation=None,
                                  name='value_out', trainable=trainable)
    if dueling_type == 'naive':
        out = value_layer + adv_layer
    elif dueling_type == 'mean':
        out = value_layer + (adv_layer - tf.reduce_mean(adv_layer, 1, keepdims=True))
    elif dueling_type == 'max':
        out = value_layer + (adv_layer - tf.reduce_max(adv_layer, 1, keepdims=True))
    else:
        raise ValueError("Unknown dueling type '%s'. Available: 'naive', 'mean', 'max'."
                         % dueling_type)
    return out
