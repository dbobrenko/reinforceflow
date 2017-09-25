from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from reinforceflow.nets import AbstractFactory, AbstractModel
from reinforceflow.core import Tuple


class DQNFactory(AbstractFactory):
    """Factory for DQN Model.
    See `DQNModel`.
    """
    def make(self, input_space, output_space, trainable=True):
        return DQNModel(input_space, output_space, trainable)


class DQNModel(AbstractModel):
    """Deep Q-Network model.
    See "Human-level control through deep reinforcement learning", Mnih et al., 2015.

    Args:
        input_space: (core.spaces.space) Observation space.
        output_space (core.spaces.space) Action space.
    """
    def __init__(self, input_space, output_space, trainable=True):
        if isinstance(input_space, Tuple) or isinstance(output_space, Tuple):
            raise ValueError('For tuple action and observation spaces '
                             'consider implementing custom network architecture.')
        self._input_ph = tf.placeholder('float32', shape=[None, *input_space.shape], name='inputs')

        net, end_points = make_dqn_body(self.input_ph, trainable)
        net = layers.fully_connected(net, num_outputs=512, activation_fn=tf.nn.relu,
                                     scope='fc1', trainable=trainable)
        end_points['fc1'] = net
        end_points['out'] = layers.fully_connected(net,
                                                   num_outputs=output_space.shape[0],
                                                   activation_fn=None, scope='out',
                                                   trainable=trainable)
        self.end_points = end_points

    @property
    def input_ph(self):
        return self._input_ph

    @property
    def output(self):
        return self.end_points['out']


def make_dqn_body(input_layer, trainable=True):
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
