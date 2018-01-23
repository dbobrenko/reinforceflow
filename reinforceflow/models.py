from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from reinforceflow.core import Continuous
from reinforceflow.utils import torch_like_initializer
xavier_initializer = layers.xavier_initializer


class Model(object):
    def build(self, input_space, output_space, trainable=True):
        """Complies model.
        Args:
            input_space: (core.spaces.space) Observation space.
            output_space (core.spaces.space) Action space.
            trainable (bool): Whether the model is trainable.

        Returns (dict):
            Dict with model endpoints: "in", "out", etc.
        """
        raise NotImplementedError


class FullyConnected(Model):
    """Fully Connected Model."""
    def __init__(self, layer_sizes=(512, 512), output_activation=None):
        self.layer_sizes = layer_sizes
        self.output_activation = output_activation

    def build(self, input_space, output_space, trainable=True):
        input_ph = tf.placeholder('float32', shape=[None] + list(input_space.shape), name='inputs')
        net = input_ph
        for i, units in enumerate(self.layer_sizes):
            net = layers.fully_connected(net, num_outputs=units, activation_fn=tf.nn.relu,
                                         trainable=trainable, scope='fc%d' % i)
        out = layers.fully_connected(net, num_outputs=output_space.shape[0],
                                     activation_fn=self.output_activation,
                                     trainable=trainable, scope='out')
        return {'in': input_ph, 'out': out}


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

    def build(self, input_space, output_space, trainable=True):
        input_ph = tf.placeholder('float32', shape=[None] + list(input_space.shape),
                                  name='inputs')
        net = layers.conv2d(inputs=input_ph,
                            num_outputs=32 if self.nature_arch else 16,
                            kernel_size=[8, 8],
                            stride=[4, 4],
                            activation_fn=tf.nn.relu,
                            padding="same",
                            scope="conv1",
                            trainable=trainable)
        net = layers.conv2d(inputs=net,
                            num_outputs=64 if self.nature_arch else 32,
                            kernel_size=[4, 4],
                            stride=[2, 2],
                            activation_fn=tf.nn.relu,
                            padding="same",
                            scope="conv2",
                            trainable=trainable)
        if self.nature_arch:
            net = layers.conv2d(inputs=net,
                                num_outputs=64,
                                kernel_size=[3, 3],
                                stride=[1, 1],
                                activation_fn=tf.nn.relu,
                                padding="same",
                                scope="conv3",
                                trainable=trainable)
        net = layers.flatten(net)

        if self.dueling:
            return {'in': input_ph, 'out': dueling_header(net, output_space)}

        net = layers.fully_connected(net, num_outputs=512 if self.nature_arch else 256,
                                     activation_fn=tf.nn.relu, scope='fc1', trainable=trainable)
        out = layers.fully_connected(net,
                                     num_outputs=output_space.shape[0], activation_fn=None,
                                     scope='out', trainable=trainable)
        return {'in': input_ph, 'out': out}


class ActorCriticConv(Model):
    """Actor-Critic Convolutional model."""

    def build(self, input_space, output_space, trainable=True):
        input_ph = tf.placeholder('float32', shape=[None] + list(input_space.shape),
                                  name='inputs')
        net = layers.conv2d(inputs=input_ph,
                            num_outputs=16,
                            kernel_size=[8, 8],
                            stride=[4, 4],
                            activation_fn=tf.nn.relu,
                            padding="same",
                            scope="conv1",
                            trainable=trainable,
                            # weights_initializer=torch_like_initializer(),
                            # biases_initializer=torch_like_initializer(),
                            weights_initializer=layers.xavier_initializer(),
                            biases_initializer=layers.xavier_initializer()
                            )
        net = layers.conv2d(inputs=net,
                            num_outputs=32,
                            kernel_size=[4, 4],
                            stride=[2, 2],
                            activation_fn=tf.nn.relu,
                            padding="same",
                            scope="conv2",
                            trainable=trainable,
                            # weights_initializer=torch_like_initializer(),
                            # biases_initializer=torch_like_initializer(),
                            weights_initializer=layers.xavier_initializer(),
                            biases_initializer=layers.xavier_initializer()
                            )
        net = layers.flatten(net)
        fc1 = layers.fully_connected(net,
                                     num_outputs=256,
                                     activation_fn=tf.nn.relu,
                                     scope='fc1',
                                     trainable=trainable,
                                     # weights_initializer=torch_like_initializer(),
                                     # biases_initializer=torch_like_initializer(),
                                     weights_initializer=layers.xavier_initializer(),
                                     biases_initializer=layers.xavier_initializer()
                                     )
        gaussian = tf.random_normal_initializer
        v = layers.fully_connected(fc1, num_outputs=1,
                                   activation_fn=None,
                                   # weights_initializer=torch_like_initializer(),
                                   # biases_initializer=torch_like_initializer(),
                                   weights_initializer=layers.xavier_initializer(),
                                   biases_initializer=layers.xavier_initializer(),
                                   scope='out_value',
                                   trainable=trainable)
        out_v = tf.squeeze(v)
        out_pi = policy_head(net, output_space, trainable)
        return {'in': input_ph, 'out': out_pi, 'out_pi': out_pi, 'out_v': out_v}


class ActorCriticFC(Model):
    """Actor-Critic Fully Connected model."""
    def __init__(self, fc_layers=(512, 512, 512)):
        self.layer_sizes = fc_layers

    def build(self, input_space, output_space, trainable=True):
        input_ph = tf.placeholder('float32', shape=[None] + list(input_space.shape),
                                  name='inputs')
        net = layers.flatten(input_ph)
        for i, units in enumerate(self.layer_sizes):
            net = layers.fully_connected(net, num_outputs=units, activation_fn=tf.nn.relu,
                                         trainable=trainable, scope='fc%d' % i)
        gaussian = tf.random_normal_initializer
        v = layers.fully_connected(net, num_outputs=1,
                                   activation_fn=None,
                                   weights_initializer=layers.xavier_initializer(),
                                   biases_initializer=layers.xavier_initializer(),
                                   scope='out_value',
                                   trainable=trainable)
        out_v = tf.squeeze(v)
        out_pi = policy_head(net, output_space, trainable)
        return {'in': input_ph, 'out': out_pi, 'out_pi': out_pi, 'out_v': out_v}


def policy_head(input_layer, output_space, trainable=True):
    gaussian = tf.random_normal_initializer
    if not isinstance(output_space, Continuous):
        pi = layers.fully_connected(input_layer, num_outputs=output_space.shape[0],
                                    activation_fn=tf.nn.softmax,
                                    weights_initializer=layers.xavier_initializer(),
                                    biases_initializer=layers.xavier_initializer(),
                                    # weights_initializer=torch_like_initializer(),
                                    # biases_initializer=torch_like_initializer(),
                                    scope='out_policy',
                                    trainable=trainable)
        return pi

    mean = layers.fully_connected(input_layer, num_outputs=output_space.shape[0],
                                  activation=tf.nn.tanh,
                                  weights_initializer=layers.xavier_initializer(),
                                  biases_initializer=layers.xavier_initializer(),
                                  scope='mean',
                                  trainable=trainable)
    std = layers.fully_connected(input_layer, num_outputs=output_space.shape[0],
                                 activation=tf.nn.softplus,
                                 weights_initializer=layers.xavier_initializer(),
                                 biases_initializer=layers.xavier_initializer(),
                                 scope='std',
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
        adv_layer = layers.fully_connected(adv_layer, num_outputs=units,
                                           activation_fn=tf.nn.relu, trainable=trainable,
                                           scope='advantage%d' % i)
    adv_layer = layers.fully_connected(adv_layer, num_outputs=output_space.shape[0],
                                       activation_fn=None, scope='adv_out', trainable=trainable)

    value_layer = input_layer
    for i, units in enumerate(value_layers):
        value_layer = layers.fully_connected(value_layer, num_outputs=units,
                                             activation_fn=tf.nn.relu, trainable=trainable,
                                             scope='value%d' % i)
    value_layer = layers.fully_connected(value_layer, num_outputs=1, activation_fn=None,
                                         scope='value_out', trainable=trainable)
    if dueling_type == 'naive':
        out = value_layer + adv_layer
    elif dueling_type == 'mean':
        out = value_layer + (adv_layer - tf.reduce_mean(adv_layer, 1, keep_dims=True))
    elif dueling_type == 'max':
        out = value_layer + (adv_layer - tf.reduce_max(adv_layer, 1, keep_dims=True))
    else:
        raise ValueError("Unknown dueling type '%s'. Available: 'naive', 'mean', 'max'."
                         % dueling_type)
    return out
