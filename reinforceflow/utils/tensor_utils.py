from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

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
    from reinforceflow.envs.gym_wrapper import ObservationStackWrap, ImageWrap
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
        for obs_id in range(all_wrappers[ObservationStackWrap].obs_stack):
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
                  q_values=None, log_performance=True, reset_stats=True, scope=''):
        """Prints passed logs, and generates TensorFlow Summary.

        Args:
            rewards (utils.RewardStats): On-policy reward incremental average.
                To disable reward logging, pass None.
            test_rewards (utils.RewardStats): Greedy-policy reward incremental average.
                To disable test reward logging, pass None.
            ep_counter (int): Episode counter.
            step_counter (int): Optimizer update step counter.
            obs_counter (int): Observation counter. To disable performance logging, pass None.
            q_values (utils.RewardStats): On-policy max Q-values incremental average.
                Used in DQN-like agents. To disable Q-values logging, pass None.
            log_performance (bool): Enables performance logging.
            reset_stats (bool): If enabled, resets passed stats counters.
            scope (str): Agent's name scope.

        Returns (tensorflow.Summary):
            TensorFlow summary logs.
        """
        value = tf.Summary.Value
        logs = []
        print_info = ''
        if rewards:
            step_av = rewards.step_average()
            episode_av = rewards.episode_average()
            print_info += "Av.Episode R: %.2f. " % episode_av
            print_info += "Av.Step R: %.2f. " % step_av
            logs += [value(tag=scope+'av_step_R', simple_value=step_av),
                     value(tag=scope+'av_ep_R', simple_value=episode_av)]
            if reset_stats:
                rewards.reset()

        if q_values:
            avg_q = q_values.step_average()
            print_info += "Av.Q: %.2f. " % avg_q
            logs += [value(tag=scope+'av_Q', simple_value=avg_q)]
            if reset_stats:
                q_values.reset()

        if test_rewards:
            test_step_av = test_rewards.step_average()
            test_episode_av = test_rewards.episode_average()
            print_info += "Greedy Av.Episode R: %.2f. " % test_rewards.episode_average()
            print_info += "Greedy Av.Step R: %.2f. " % test_rewards.step_average()
            logs += [value(tag=scope+'av_greedy_step_R', simple_value=test_step_av),
                     value(tag=scope+'av_greedy_ep_R', simple_value=test_episode_av)]
            if reset_stats:
                test_rewards.reset()

        if print_info:
            name = scope.replace('/', '') + '. ' if len(scope) else scope
            print_info = "%s%sObs: %d. Update step: %d. Ep: %d" \
                         % (name, print_info, obs_counter, step_counter, ep_counter)
            logger.info(print_info)

        if log_performance:
            step_per_sec = (step_counter - self.last_step) / (time.time() - self.last_time)
            obs_per_sec = (obs_counter - self.last_obs) / (time.time() - self.last_time)
            logger.info("Obs/sec: %0.2f. Optimizer update/sec: %0.2f."
                        % (obs_per_sec, step_per_sec))
            logs += [value(tag=scope+'total_ep', simple_value=ep_counter),
                     value(tag=scope+'step_per_sec', simple_value=step_per_sec),
                     value(tag=scope+'obs_per_sec', simple_value=obs_per_sec),
                     ]
            self.last_step = step_counter
            self.last_obs = obs_counter
            self.last_time = time.time()
        return tf.Summary(value=logs)
