from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf

from reinforceflow import logger


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
