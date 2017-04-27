from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import six
from scipy.signal import lfilter
import tensorflow as tf
from tensorflow.python.training import optimizer as base_optimizer
from tensorflow.python.framework import ops
from reinforceflow import logger


__OPTIMIZER_MAP__ = {
    'rms': tf.train.RMSPropOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'rmspropoptimizer': tf.train.RMSPropOptimizer,
    'adam': tf.train.AdamOptimizer,
    'adamoptimizer': tf.train.AdamOptimizer,
    'sgd': tf.train.GradientDescentOptimizer,
    'gradientdescent': tf.train.GradientDescentOptimizer,
    'gradientdescentoptimizer': tf.train.GradientDescentOptimizer
}


__DECAY_MAP__ = {
    'polynomial_decay': tf.train.polynomial_decay,
    'polynomial': tf.train.polynomial_decay,
    'poly': tf.train.polynomial_decay,
    'exponential_decay': tf.train.exponential_decay,
    'exponential': tf.train.exponential_decay,
    'exp': tf.train.exponential_decay
}


def get_learning_rate(optimizer):
    """Hack for getting already instantiated Optimizer's learning rate"""
    if hasattr(optimizer, '_learning_rate'):
        return optimizer._learning_rate  # pylint: disable=W0212
    elif hasattr(optimizer, '_lr'):
        return optimizer._lr  # pylint: disable=W0212
    raise ValueError("Cannot get optimizer (%s) learning rate." % optimizer)


def set_learning_rate(optimizer, learning_rate):
    if hasattr(optimizer, '_learning_rate'):
        optimizer._learning_rate = learning_rate  # pylint: disable=W0212
    elif hasattr(optimizer, '_lr'):
        optimizer._lr = learning_rate  # pylint: disable=W0212
    else:
        raise ValueError("Cannot get optimizer (%s) learning rate." % optimizer)


def create_optimizer(opt, learning_rate=None, decay=None, global_step=None, decay_poly_steps=None,
                     decay_poly_end_lr=0.0001, decay_poly_power=1.0, decay_rate=0.96):
    replace_existed_lr = False
    if isinstance(opt, base_optimizer.Optimizer):
        if learning_rate is None:
            learning_rate = get_learning_rate(opt)
            # Check for possible decay in optimizer's learning rate
            if isinstance(learning_rate, ops.Tensor) and learning_rate.get_shape().ndims == 0:
                if decay:
                    logger.warn('Passed optimizer already has learning rate of type Tensor. '
                                'Skipping learning rate decay (%s), while '
                                'leaving original learning rate Tensor (%s) unchanged.' % (decay, learning_rate))
                    return opt, learning_rate
            else:
                replace_existed_lr = True

    if learning_rate is None:
        raise ValueError("Learning rate should be specified, in order to instantiate optimizer %s." % opt)

    # Check for possible decay in learning rate
    if isinstance(learning_rate, ops.Tensor) and learning_rate.get_shape().ndims == 0:
        if decay:
            logger.warn('Passed learning rate is already of type Tensor. '
                        'Skipping learning rate decay (%s), while '
                        'leaving original learning rate Tensor (%s) unchanged.' % (decay, learning_rate))
    elif isinstance(learning_rate, (float, int)):
        if learning_rate < 0.0:
            raise ValueError("Learning rate should be >= 0. Got: %s.", learning_rate)
        learning_rate = ops.convert_to_tensor(learning_rate, dtype=tf.float32, name="learning_rate")
        if decay:
            if global_step is None:
                raise ValueError('Global step should be specified, in order to use learning rate decay.')
            learning_rate = create_decay(decay, global_step, learning_rate, decay_poly_steps, decay_poly_end_lr,
                                         decay_poly_power, decay_rate)
            if replace_existed_lr:
                set_learning_rate(opt, learning_rate)
                return opt, learning_rate
    else:
        raise ValueError("Learning rate should be 0d Tensor or float. "
                         "Got %s of type %s" % (str(learning_rate), str(type(learning_rate))))

    if isinstance(opt, type) and issubclass(opt, base_optimizer.Optimizer):
        opt = opt(learning_rate=learning_rate)
    elif isinstance(opt, six.string_types):
        opt = opt.lower()
        if opt not in __OPTIMIZER_MAP__:
            raise ValueError("Unknown optimizer name %s. Available: %s." % (opt, ', '.join(__OPTIMIZER_MAP__)))
        opt = __OPTIMIZER_MAP__[opt](learning_rate=learning_rate)
    else:
        raise ValueError("Unknown optimizer %s. Should be either a class name string, subclass of Optimizer or "
                         "instance of Optimizer." % str(opt))
    return opt, learning_rate


def create_decay(decay, global_step, learning_rate=None, decay_steps=None, end_learning_rate=0.0001, power=1.0,
                 decay_rate=0.96):
    if callable(decay) and hasattr(decay, __name__):
        decay = decay.__name__
    if isinstance(decay, six.string_types):
        decay = decay.lower()
        if decay in ['polynomial_decay', 'polynomial', 'poly']:
            if learning_rate is None or decay_steps is None:
                raise ValueError('You should specify learning rate and decay steps for %s function'
                                 % tf.train.polynomial_decay)
            learning_rate = tf.train.polynomial_decay(learning_rate, global_step, decay_steps,
                                                      end_learning_rate=end_learning_rate, power=power)
        elif decay in ['exponential_decay', 'exponential', 'exp']:
            learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate=decay_rate)
        else:
            raise ValueError('Unknown decay function %s. Available: %s' % (decay, ', '.join(__DECAY_MAP__)))
    else:
        raise ValueError('Decay should be either a decay function or a function name string.')
    return learning_rate


class AgentSummary(object):
    # TODO: add histograms and images
    def __init__(self, sess, logdir, *variables, scalar_tags=[]):
        self.sess = sess

        self.writer = tf.summary.FileWriter(logdir, sess.graph)
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        for variable in variables:
            summaries.add(tf.summary.histogram('weights/%s' % variable.op.name, variable))
            summaries.add(tf.summary.scalar('weights_sparsity/%s' % variable.op.name, tf.nn.zero_fraction(variable)))

        self.scalar_ph = {}
        for key in scalar_tags:
            self.scalar_ph[key] = tf.placeholder(dtype='float32')
            summaries.add(tf.summary.scalar(key, self.scalar_ph[key]))

        # self.histogram_ph = {}
        # for key in histogram_tags:
        #     self.histogram_ph[key] = tf.placeholder(dtype='float32', shape=[None, None], name=key)
        #     summaries.add(tf.summary.histogram(key, self.scalar_ph[key]))
        self.summary_op = tf.summary.merge(list(summaries), name='summary_op')

    def write_summary(self, step, sess=None, summary_dict={}):
        self.sess = sess or self.sess
        summary_vars = self.sess.run(self.summary_op, {self.scalar_ph[k]: v for k, v in summary_dict.items()})
        self.writer.add_summary(summary_vars, global_step=step)


def discount_rewards(rewards, gamma):
    return lfilter([1], [1, -gamma], rewards[::-1])[::-1]