from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import gym
from reinforceflow.envs.env_wrapper import EnvWrapper
from reinforceflow.envs.gym_wrapper import GymWrapper
from reinforceflow.envs.gym_wrapper import GymPixelWrapper
from reinforceflow import logger


class EnvFactory(object):
    @staticmethod
    def make(env,
             obs_stack=1,
             action_repeat=1,
             pixel_env=False,
             random_start=0,
             resize_width=None,
             resize_height=None,
             use_smart_wrap=True):
        """Wraps environment into `reinforceflow.envs.EnvWrapper`.

        Args:
            env: Environment instance.
            obs_stack: (int) Number of previous frames to stack to the current.
                             Used for the agents without short-term sumtree.
            action_repeat (int): Number of frames to skip with last action repeated.
            pixel_env: (bool) Whether environment has pixel screen observations.
            resize_width: (int) Resize width. Applies only for pixel screen environments.
            resize_height: (int) Resize height. Applies only for pixel screen environments.
            random_start (int): (TODO) Determines the max number of randomly skipped frames,
                                see Mnih et al., 2015.
            use_smart_wrap (bool): Enable smart wrapping. E.g.:
                           Atari environments will be processed as stated in Mnih et al., 2015.

        Returns (gym.Wrapper): Environment instance.
        """
        if isinstance(env, six.string_types):
            env = gym.make(env)

        if use_smart_wrap:
            if hasattr(env.env, 'ale'):
                logger.info('Creating Atari Gym environment wrapper.')
                return GymPixelWrapper(env=env,
                                       action_repeat=action_repeat or 4,
                                       obs_stack=obs_stack or 4,
                                       resize_width=84,
                                       resize_height=84)
        if pixel_env:
            logger.info('Creating Gym pixel environment wrapper.')
            return GymPixelWrapper(env=env,
                                   action_repeat=action_repeat,
                                   obs_stack=obs_stack,
                                   resize_width=resize_width,
                                   resize_height=resize_height)

        logger.info('Creating Gym environment wrapper.')
        return GymWrapper(env=env, action_repeat=action_repeat, obs_stack=obs_stack)
