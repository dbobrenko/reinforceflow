from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from reinforceflow.utils import utils

_EPSILON = 1e-8


class LossCompositor(object):
    def __init__(self, losses=None):
        self.losses = [] if losses is None else losses

    def add(self, loss):
        if isinstance(loss, (list, set)):
            self.losses += list(loss)
        else:
            self.losses.append(loss)

    def loss(self, endpoints, action, reward, term):
        value = tf.constant(0.0)
        for l in self.losses:
            value += l.loss(endpoints=endpoints, action=action, reward=reward, term=term)
        return value


class BaseLoss(object):
    def __init__(self, coef=1.0, name='loss'):
        self.coef = coef
        self.name = name

    def loss(self, endpoints, action, name, **kwargs):
        if not isinstance(endpoints, dict):
            raise ValueError("Network output must be a dict, got %s." % type(endpoints))


class PolicyGradientLoss(BaseLoss):
    def loss(self, endpoints, action, reward, name='pg', **kwargs):
        """Policy Gradient loss.

        Args:
            endpoints (dict): Dict with network endpoints.
                Must contain "policy" - action probabilities, "value" - baseline.
            action (Placeholder): Vectorized action placeholder.
            name (str): Loss name.

        Returns (Tensor):
            Policy Gradient loss operation.
        """
        super(PolicyGradientLoss, self).loss(endpoints, action, name)
        if 'policy' not in endpoints or 'value' not in endpoints:
            raise ValueError("Network output must contain policy and value fields, "
                             "in order to use %s. Got %s" % (self.__class__.__name__, endpoints))
        logprob = -tf.log(endpoints['policy'] + _EPSILON)
        baseline = tf.stop_gradient(reward - endpoints['value'])
        crossentropy = tf.reduce_sum(logprob * action, 1)
        pg = tf.reduce_sum(crossentropy * baseline)
        # pg = -tf.reduce_sum(tf.reduce_sum(logprob * action, 1) * baseline, name=self.name)
        return self.coef * pg


class EntropyLoss(BaseLoss):
    def __init__(self, coef=-0.01):
        super(EntropyLoss, self).__init__(coef=coef)

    def loss(self, endpoints, action, **kwargs):
        if 'policy' not in endpoints:
            raise ValueError("Network output must contain policy field, "
                             "in order to use %s. Got %s" % (self.__class__.__name__, endpoints))
        logprob = tf.log(endpoints['policy'] + _EPSILON)
        entropy = -tf.reduce_sum(endpoints['policy'] * logprob)
        return self.coef * entropy


class AdvantageLoss(BaseLoss):
    def loss(self, endpoints, reward, **kwargs):
        if 'value' not in endpoints:
            raise ValueError("Network output must contain value field, "
                             "in order to use %s. Got %s" % (self.__class__.__name__, endpoints))
        return self.coef * tf.subtract(reward, endpoints['value'], name=self.name)


class QLoss(BaseLoss):
    """Action-Value Temporal-Difference error. See `td_error`."""
    def __init__(self, coef=1.0, importance_sampling=None):
        super(QLoss, self).__init__(coef=coef)
        self.importance_sampling = importance_sampling

    def loss(self, endpoints, reward, action, **kwargs):
        if 'value' not in endpoints:
            raise ValueError("Network output must contain value field, "
                             "in order to use %s. Got %s" % (self.__class__.__name__, endpoints))
        td = reward - tf.reduce_sum(endpoints['value'] * action, 1)
        if self.importance_sampling is not None:
            td *= self.importance_sampling
        return self.coef * tf.reduce_mean(tf.square(td), name=self.name)


class TDLoss(BaseLoss):
    def __init__(self, coef=1.0, importance_sampling=None):
        super(TDLoss, self).__init__(coef=coef)
        self.importance_sampling = importance_sampling

    def loss(self, endpoints, reward, importance_sampling=None, **kwargs):
        if 'value' not in endpoints:
            raise ValueError("Network output must contain value field, "
                             "in order to use %s. Got %s" % (self.__class__.__name__, endpoints))
        td = reward - endpoints['value']
        if importance_sampling is not None:
            td *= importance_sampling
        return self.coef * tf.reduce_mean(tf.square(td), name=self.name)


# UNREAL Loss
class PixelControlLoss(BaseLoss):
    def __init__(self, control_layers, num_actions, gridsize=(20, 20), fc_units=(9*9*32,),
                 deconv_kernel=(4, 4), deconv_strides=(2, 2), coef=1.0):
        super(PixelControlLoss, self).__init__(coef)
        self.control_layers = control_layers if utils.isarray(control_layers) else [control_layers]
        self.fc_units = fc_units if utils.isarray(fc_units) else [fc_units]
        self.num_actions = num_actions
        self.gridsize = gridsize
        self.deconv_kernel = deconv_kernel
        self.deconv_strides = deconv_strides

    def loss(self, endpoints, reward, importance_sampling=None, **kwargs):
        assert len(self.control_layers) == len(self.fc_units),\
            "Number of control layers must be equal to the number of FC (%d != %d)."\
            % (len(self.control_layers), len(self.fc_units))

        for layer, units in zip(self.control_layers, self.fc_units):
            flatten = tf.layers.flatten(layer)
            fc = tf.layers.dense(flatten, units=units, activation=tf.nn.relu)
            adv = tf.layers.conv2d_transpose(inputs=fc,
                                             filters=self.num_actions,
                                             kernel_size=self.deconv_kernel,
                                             strides=self.deconv_strides)
            value = tf.layers.conv2d_transpose(inputs=fc,
                                               filters=self.num_actions,
                                               kernel_size=self.deconv_kernel,
                                               strides=self.deconv_strides)
            # q = value + (adv - tf.reduce_mean(adv, 1, keepdims=True))
            # TODO
