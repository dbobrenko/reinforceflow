from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from reinforceflow.core import Continious
from reinforceflow.nets import AbstractFactory, AbstractModel, make_dqn_body
from reinforceflow.core import Tuple


class A3CMLPFactory(AbstractFactory):
    """Factory for Multilayer Perceptron."""
    def __init__(self, layer_sizes=(512, 512, 512), policy_activation=tf.nn.softmax):
        self.layer_sizes = layer_sizes
        self.policy_activation = policy_activation

    def make(self, input_space, output_space, trainable=True):
        return A3CMLP(input_space, output_space, layer_sizes=self.layer_sizes, trainable=trainable)


class A3CConvFactory(AbstractFactory):
    def __init__(self, policy_activation=tf.nn.softmax):
        self.policy_activation = policy_activation

    def make(self, input_space, output_space, trainable=True):
        return A3CConv(input_space, output_space, trainable=trainable)


class A3CConv(AbstractModel):
    """Asynchronous Advantage Actor-Critic Conv + Feed-Forward model."""
    def __init__(self, input_space, output_space, trainable=True):
        if isinstance(input_space, Tuple) or isinstance(output_space, Tuple):
            raise ValueError('For tuple action and observation spaces '
                             'consider implementing custom network architecture.')
        self._input_ph = tf.placeholder('float32', shape=[None] + list(input_space.shape), name='inputs')
        net, end_points = make_dqn_body(self.input_ph, trainable)
        end_points['fc1'] = layers.fully_connected(net, num_outputs=512, activation_fn=tf.nn.relu,
                                                   scope='fc1', trainable=trainable)
        gaussian = tf.random_normal_initializer
        v = layers.fully_connected(end_points['fc1'], num_outputs=1,
                                   activation_fn=None,
                                   weights_initializer=gaussian(0.0, 0.1),
                                   biases_initializer=gaussian(0.05, 0.1),
                                   scope='out_value',
                                   trainable=trainable)
        end_points['out_value'] = tf.squeeze(v)
        header_endpoints = make_a3c_header(net, input_space, output_space, trainable)
        end_points.update(header_endpoints)
        self.end_points = end_points
        self.output_policy = self.output

    @property
    def input_ph(self):
        return self._input_ph

    @property
    def output(self):
        return self.end_points['out_policy']

    @property
    def output_value(self):
        return self.end_points['out_value']


class A3CMLP(AbstractModel):
    """Asynchronous Advantage Actor-Critic MLP model."""
    def __init__(self, input_space, output_space, layer_sizes=(512, 512, 512), trainable=True):
        if isinstance(input_space, Tuple) or isinstance(output_space, Tuple):
            raise ValueError('For tuple action and observation spaces '
                             'consider implementing custom network architecture.')
        self._input_ph = tf.placeholder('float32', shape=[None] + list(input_space.shape),
                                        name='inputs')
        end_points = {}
        net = layers.flatten(self._input_ph)
        for i, units in enumerate(layer_sizes):
            name = 'fc%d' % i
            net = layers.fully_connected(net, num_outputs=units, activation_fn=tf.nn.relu,
                                         trainable=trainable, scope=name)
            end_points[name] = net
        gaussian = tf.random_normal_initializer
        v = layers.fully_connected(net, num_outputs=1,
                                   activation_fn=None,
                                   weights_initializer=gaussian(0.0, 0.1),
                                   biases_initializer=gaussian(0.05, 0.1),
                                   scope='out_value',
                                   trainable=trainable)
        end_points['out_value'] = tf.squeeze(v)
        header_endpoints = make_a3c_header(net, input_space, output_space, trainable)
        end_points.update(header_endpoints)
        self.end_points = end_points
        self.output_policy = self.output

    @property
    def input_ph(self):
        return self._input_ph

    @property
    def output(self):
        return self.end_points['out_policy']

    @property
    def output_value(self):
        return self.end_points['out_value']


def make_a3c_header(input_layer, input_space, output_space, trainable=True):
    end_points = {}
    gaussian = tf.random_normal_initializer
    if isinstance(input_space, Continious):
        p = layers.fully_connected(input_layer, num_outputs=output_space.shape[0],
                                   activation_fn=tf.nn.softmax,
                                   weights_initializer=gaussian(0.0, 0.1),
                                   biases_initializer=gaussian(0.05, 0.1),
                                   scope='out_policy',
                                   trainable=trainable)
        end_points['out_policy'] = p
    else:
        mu = layers.fully_connected(input_layer, num_outputs=output_space.shape[0],
                                    activation=tf.nn.tanh,
                                    weights_initializer=gaussian(0.0, 0.1),
                                    biases_initializer=gaussian(0.05, 0.1),
                                    scope='mu',
                                    trainable=trainable)
        sigma = layers.fully_connected(input_layer, num_outputs=output_space.shape[0],
                                       activation=tf.nn.softplus,
                                       weights_initializer=gaussian(0.0, 0.1),
                                       biases_initializer=gaussian(0.05, 0.1),
                                       scope='sigma',
                                       trainable=trainable)
        mu = tf.squeeze(mu)
        end_points['mu'] = mu
        sigma = tf.squeeze(sigma)
        end_points['sigma'] = sigma
        dist = tf.contrib.distributions.Normal(mu, sigma)
        end_points['dist'] = dist
        end_points['out_policy'] = tf.clip_by_value(dist.sample(1),
                                                    input_space.low, input_space.high)
    return end_points
