from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf

from reinforceflow.utils import utils
from reinforceflow import logger


def discount_trajectory_op(rewards, terms, traj_ends, gamma, ev):
    # Predict EV for bootstrap states
    # EV len should be equal to the trajectory len or bootstrap states len
    bootstrap_idx = tf.logical_xor(traj_ends, terms)
    num_bootstrap = tf.reduce_sum(tf.cast(bootstrap_idx, 'int32'))
    ev = tf.cond(tf.equal(num_bootstrap, 0),
                 lambda: tf.zeros_like(traj_ends, 'float32'),
                 lambda: ev)
    with tf.device("/cpu:0"):
        discount = tf.py_func(utils.discount_trajectory,
                              [rewards, terms, traj_ends, gamma, ev],
                              tf.float32)
    # If batch consists from randomly shuffled samples (usually used in Replay trainers)
    # discount = rewards + gamma * ev
    return discount


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
        env (gym.Env): Environment instance.
    """
    from reinforceflow.envs.wrapper import ObservationStackWrap, ImageWrap
    # Get all wrappers
    all_wrappers = {}
    env_wrapper = env
    while True:
        if isinstance(env_wrapper, gym.Wrapper):
            all_wrappers[env_wrapper.__class__] = env_wrapper
            env_wrapper = env_wrapper.env
        else:
            break

    # Check for grayscale
    gray = False
    if ImageWrap in all_wrappers:
        gray = all_wrappers[ImageWrap].grayscale

    # Check and wrap observation stack
    if ObservationStackWrap in all_wrappers:
        channels = 1 if gray else 3
        for obs_id in range(all_wrappers[ObservationStackWrap].stack_len):
            o = obs[:, :, :, obs_id*channels:(obs_id+1)*channels]
            tf.summary.image('observation%d' % obs_id, o, max_outputs=1)
        return

    # Try to wrap current observation
    if len(env.observation_space.shape) == 1:
        tf.summary.histogram('observation', obs)
    elif len(env.observation_space.shape) == 2:
        tf.summary.image('observation', obs)
    elif len(env.observation_space.shape) == 3 and env.observation_space.shape[2] in (1, 3):
        tf.summary.image('observation', obs)
    else:
        logger.warn('Cannot create summary for observation with shape',
                    env.observation_space.shape)


def torch_like_initializer():
    return tf.contrib.layers.variance_scaling_initializer(factor=1/3,
                                                          mode='FAN_IN',
                                                          uniform=True)


_INITIALIZED = set()


def initialize_variables(sess):
    """Initialize all the uninitialized variables in the global scope."""
    new_variables = set(tf.global_variables()) - _INITIALIZED
    sess.run(tf.variables_initializer(new_variables))
    _INITIALIZED.update(new_variables)
