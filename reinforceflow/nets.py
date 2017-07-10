from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


def dqn(input_shape, output_size, trainable=True):
    output_size = np.ravel(output_size)
    if len(output_size) != 1:
        raise ValueError('Output size must be scalar or rank 1 nd.array.')
    output_size = output_size[0]
    end_points = {}
    inputs = tf.placeholder('float32', shape=input_shape, name='inputs')
    end_points['inputs'] = inputs
    net = layers.conv2d(inputs=inputs,
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
                        strides=[2, 2],
                        activation=tf.nn.relu,
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
    conv3 = layers.flatten(net)
    net = layers.fully_connected(conv3, num_outputs=512, activation_fn=tf.nn.relu,
                                 scope='fc1', trainable=trainable)
    end_points['fc1'] = net
    net = layers.fully_connected(net, num_outputs=output_size, activation_fn=None,
                                 scope='out', trainable=trainable)
    end_points['outs'] = net
    return inputs, net, end_points


def mlp(input_shape, output_size, layer_sizes=(16, 16), output_activation=None, trainable=True):
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
    return inputs, net, end_points
