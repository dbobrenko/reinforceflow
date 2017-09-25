from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from reinforceflow.nets import AbstractFactory, AbstractModel
from reinforceflow.core import Tuple


class MLPFactory(AbstractFactory):
    """Factory for Multilayer Perceptron."""
    def __init__(self, layer_sizes=(512, 512, 512)):
        self.layer_sizes = layer_sizes

    def make(self, input_space, output_space, trainable=True):
        return MLPModel(input_space, output_space, layer_sizes=self.layer_sizes,
                        trainable=trainable)


class MLPModel(AbstractModel):
    """Multilayer Perceptron."""
    def __init__(self, input_space, output_space, layer_sizes=(512, 512, 512),
                 output_activation=None, trainable=True):
        if isinstance(input_space, Tuple) or isinstance(output_space, Tuple):
            raise ValueError('For tuple action and observation spaces '
                             'consider implementing custom network architecture.')
        end_points = {}
        self._input_ph = tf.placeholder('float32', shape=[None, *input_space.shape], name='inputs')
        net = self._input_ph
        for i, units in enumerate(layer_sizes):
            name = 'fc%d' % i
            net = layers.fully_connected(net, num_outputs=units, activation_fn=tf.nn.relu,
                                         trainable=trainable, scope=name)
            end_points[name] = net
        end_points['out'] = layers.fully_connected(net, num_outputs=output_space.shape[0],
                                                   activation_fn=output_activation,
                                                   trainable=trainable, scope='out')
        self.end_points = end_points

    @property
    def input_ph(self):
        return self._input_ph
    
    @property
    def output(self):
        return self.end_points['out']
