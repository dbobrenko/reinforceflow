from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from reinforceflow.envs import GymPixelWrapper
from reinforceflow import logger

_OPTIMIZER_MAP = {
    'rms': tf.train.RMSPropOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'rmspropoptimizer': tf.train.RMSPropOptimizer,
    'adam': tf.train.AdamOptimizer,
    'adamoptimizer': tf.train.AdamOptimizer,
    'sgd': tf.train.GradientDescentOptimizer,
    'gradientdescent': tf.train.GradientDescentOptimizer,
    'gradientdescentoptimizer': tf.train.GradientDescentOptimizer
}


_DECAY_MAP = {
    'polynomial_decay': tf.train.polynomial_decay,
    'polynomial': tf.train.polynomial_decay,
    'poly': tf.train.polynomial_decay,
    'exponential_decay': tf.train.exponential_decay,
    'exponential': tf.train.exponential_decay,
    'exp': tf.train.exponential_decay
}


def create_optimizer(opt, learning_rate, optimizer_args=None, decay=None,
                     decay_args=None, global_step=None):
    """Creates TensorFlow optimizer with given args and learning rate decay.

    Args:
        opt: TensorFlow optimizer, expects string or callable object.
        learning_rate (float or Tensor): Optimizer learning rate.
        optimizer_args (dict): TensorFlow optimizer kwargs.
        decay (str or function): Learning rate decay. Available: poly, exp.
        decay_args (dict): Learning rate decay kwargs.
        global_step (Tensor): Optimizer global step.

    Returns (tuple):
        Optimizer instance, learning rate tensor.
    """
    if optimizer_args is None:
        optimizer_args = {}
    # Process learning rate
    if isinstance(learning_rate, ops.Tensor) and learning_rate.get_shape().ndims == 0:
        if decay:
            logger.warn("Passed learning rate is already of type Tensor. "
                        "Leaving optimizer original learning rate Tensor (%s) unchanged."
                        % learning_rate)
    elif isinstance(learning_rate, (float, int)):
        if learning_rate < 0.0:
            raise ValueError("Learning rate must be >= 0. Got: %s.", learning_rate)
        learning_rate = ops.convert_to_tensor(learning_rate, dtype=tf.float32,
                                              name="learning_rate")
        if decay:
            if global_step is None:
                raise ValueError('Global step must be specified, '
                                 'in order to use learning rate decay.')
            if decay_args is None:
                decay_args = {}
            learning_rate = create_decay(decay, learning_rate, global_step, **decay_args)
    else:
        raise ValueError("Learning rate should be 0d Tensor or float. "
                         "Got %s of type %s" % (str(learning_rate), str(type(learning_rate))))

    # Create optimizer
    if callable(opt):
        return opt(learning_rate=learning_rate, **optimizer_args), learning_rate
    elif isinstance(opt, six.string_types):
        opt = opt.lower()
        if opt not in _OPTIMIZER_MAP:
            raise ValueError("Unknown optimizer name %s. Available: %s."
                             % (opt, ', '.join(_OPTIMIZER_MAP)))
        return _OPTIMIZER_MAP[opt](learning_rate=learning_rate, **optimizer_args), learning_rate
    else:
        raise ValueError("Unknown optimizer %s. Should be either a class name string,"
                         "subclass of Optimizer or any callable object"
                         "with `learning_rate` argument." % str(opt))


def create_decay(decay, learning_rate, global_step, **kwargs):
    """Creates learning rate decay with given args.

    Args:
        decay (str): Learning rate decay. Available: poly, exp.
        learning_rate (float or Tensor): Optimizer learning rate.
        global_step (Tensor): Optimizer global step.
        **kwargs: Learning rate decay function kwargs.

    Returns (Tensor):
        Learning rate with applied decay.
    """
    if callable(decay) and hasattr(decay, __name__):
        decay = decay.__name__

    if isinstance(decay, six.string_types):
        decay = decay.lower()
        if decay in ['polynomial_decay', 'polynomial', 'poly']:
            if 'decay_steps' not in kwargs:
                raise ValueError('You should specify decay_steps argument for the `%s`'
                                 ' decay function.' % decay)
            learning_rate = tf.train.polynomial_decay(learning_rate, global_step, **kwargs)
        elif decay in ['exponential_decay', 'exponential', 'exp']:
            learning_rate = tf.train.exponential_decay(learning_rate, global_step, **kwargs)
        else:
            raise ValueError('Unknown decay function %s. Available: %s'
                             % (decay, ', '.join(_DECAY_MAP)))
    else:
        raise ValueError('Decay should be either a decay function or a function name string.')
    return learning_rate


def add_grads_summary(grad_vars):
    """Adds summary for weights and gradients.

    Args:
        grad_vars (list): List of (gradients, weights) tensors.
    """
    for grad, w in grad_vars:
        tf.summary.histogram(w.name, w)
        if grad is not None:
            tf.summary.histogram(w.name + '/gradients', grad)


def add_observation_summary(obs, env):
    """Adds observation summary.
    Supports observation tensors with 1, 2 and 3 dimensions only.
    1-D tensors logs as histogram summary.
    2-D and 3-D tensors logs as image summary.

    Args:
        obs (Tensor): Observation.
        env (envs.Env): Environment instance.
    """
    if env.obs_stack == 1:
        if len(env.obs_space.shape) == 1:
            tf.summary.histogram('observation', obs)
        elif len(env.obs_space.shape) == 2:
            tf.summary.image('observation', obs)
        elif len(env.obs_space.shape) == 3 and env.obs_space.shape[2] in (1, 3):
            tf.summary.image('observation', obs)
    elif isinstance(env, GymPixelWrapper):
        channels = 1 if env.is_grayscale else 3
        for obs_id in range(env.obs_stack):
            o = obs[:, :, :, obs_id*channels:(obs_id+1)*channels]
            tf.summary.image('observation%d' % obs_id, o, max_outputs=1)
    else:
        logger.warn('Cannot create summary for observation with shape', env.obs_space.shape)


class SummaryLogger(object):
    def __init__(self, step_counter, obs_counter):
        """Agent's performance logger.

        Args:
            step_counter (int): Initial optimizer update step.
            obs_counter (int): Initial observation counter.
        """
        self.last_time = time.time()
        self.last_step = step_counter
        self.last_obs = obs_counter

    def summarize(self, rewards, test_rewards, ep_counter, step_counter, obs_counter,
                  q_values=None, log_performance=True, reset_rewards=True, scope=''):
        """

        Args:
            rewards (utils.IncrementalAverage): On-policy reward incremental average.
                To disable reward logging, pass None.
            test_rewards (utils.IncrementalAverage): Greedy-policy reward incremental average.
                To disable test reward logging, pass None.
            ep_counter (int): Episode counter.
            step_counter (int): Optimizer update step counter.
            obs_counter (int): Observation counter. To disable performance logging, pass None.
            q_values (utils.IncrementalAverage): On-policy max Q-values incremental average.
                Used in DQN-like agents. To disable Q-values logging, pass None.
            log_performance (bool): Enables performance logging.
            reset_rewards (bool): If enabled, resets `rewards` and `test_rewards` counters.
            scope (str): Agent's name scope.

        Returns:
            TensorFlow summary logs.
        """
        logs = []
        print_info = ''
        if rewards:
            num_ep = rewards.length
            max_r = rewards.max
            min_r = rewards.min
            avg_r = rewards.reset() if reset_rewards else rewards.compute_average()
            print_info += "On-policy Avg R: %.2f. " % avg_r
            logs += [tf.Summary.Value(tag=scope + 'metrics/num_ep', simple_value=num_ep),
                     tf.Summary.Value(tag=scope + 'metrics/max_R', simple_value=max_r),
                     tf.Summary.Value(tag=scope + 'metrics/min_R', simple_value=min_r),
                     tf.Summary.Value(tag=scope + 'metrics/avg_R', simple_value=avg_r)]
        if q_values:
            max_q = q_values.max
            min_q = q_values.min
            avg_q = q_values.reset() if reset_rewards else q_values.compute_average()
            print_info += "Avg Q-value: %.2f. " % avg_q
            logs += [tf.Summary.Value(tag=scope + 'metrics/avg_Q', simple_value=avg_q),
                     tf.Summary.Value(tag=scope + 'metrics/max_Q', simple_value=max_q),
                     tf.Summary.Value(tag=scope + 'metrics/min_Q', simple_value=min_q)]
        if test_rewards:
            test_max_r = test_rewards.max
            test_min_r = test_rewards.min
            test_avg_r = test_rewards.reset() if reset_rewards else test_rewards.compute_average()
            print_info += "Greedy-policy Avg R: %.2f. " % test_avg_r
            logs += [tf.Summary.Value(tag=scope + 'metrics/avg_greedy_R', simple_value=test_avg_r),
                     tf.Summary.Value(tag=scope + 'metrics/max_greedy_R', simple_value=test_max_r),
                     tf.Summary.Value(tag=scope + 'metrics/min_greedy_R', simple_value=test_min_r)]
        if print_info:
            name = scope.replace('/', '') + '. ' if len(scope) else scope
            print_info = "%s%sObs: %d. Update step: %d. Ep: %d" \
                         % (name, print_info, obs_counter, step_counter, ep_counter)
            logger.info(print_info)
        if log_performance:
            step_per_sec = (step_counter - self.last_step) / (time.time() - self.last_time)
            obs_per_sec = (obs_counter - self.last_obs) / (time.time() - self.last_time)
            logger.info("Observation/sec: %0.2f. Optimizer update/sec: %0.2f."
                        % (obs_per_sec, step_per_sec))
            logs += [tf.Summary.Value(tag=scope + 'metrics/total_ep', simple_value=ep_counter),
                     tf.Summary.Value(tag=scope + 'step_per_sec', simple_value=step_per_sec),
                     tf.Summary.Value(tag=scope + 'obs_per_sec', simple_value=obs_per_sec),
                     ]
            self.last_step = step_counter
            self.last_obs = obs_counter
            self.last_time = time.time()

        return tf.Summary(value=logs)
