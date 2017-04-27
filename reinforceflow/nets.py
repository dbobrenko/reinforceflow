from __future__ import absolute_import

import tensorflow as tf


def dqn(input_shape, output_size):
    end_points = {}
    inputs = tf.placeholder('float32', shape=input_shape, name='inputs')
    end_points['inputs'] = inputs
    net = tf.layers.conv2d(inputs=inputs,
                           filters=32,
                           kernel_size=[8, 8],
                           strides=[4, 4],
                           activation=tf.nn.relu,
                           padding="same",
                           name="conv1")
    end_points['conv1'] = net
    net = tf.layers.conv2d(inputs=net,
                           filters=64,
                           kernel_size=[4, 4],
                           strides=[2, 2],
                           activation=tf.nn.relu,
                           padding="same",
                           name="conv2")
    end_points['conv2'] = net
    net = tf.layers.conv2d(inputs=net,
                           filters=64,
                           kernel_size=[3, 3],
                           strides=[1, 1],
                           activation=tf.nn.relu,
                           padding="same",
                           name="conv3")
    end_points['conv3'] = net
    conv3 = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(conv3, units=512, activation=tf.nn.relu, name='fc1')
    end_points['fc1'] = net
    net = tf.layers.dense(net, units=output_size, activation=None)
    end_points['outs'] = net
    return inputs, net, end_points


def mlp(input_shape, output_size, layers=(16, 16, 16), output_activation=None):
    end_points = {}
    inputs = tf.placeholder('float32', shape=input_shape, name='inputs')
    end_points['inputs'] = inputs
    net = tf.contrib.layers.flatten(inputs)
    for i, units in enumerate(layers):
        name = 'fc%d' % i
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu, name=name)
        end_points[name] = net
    net = tf.layers.dense(net, units=output_size, activation=output_activation, name='outs')
    end_points['outs'] = net
    return inputs, net, end_points
