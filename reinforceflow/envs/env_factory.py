from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import gym
from reinforceflow.envs.env_wrappers import EnvWrapper
from reinforceflow.envs.env_wrappers import AtariWrapper
from reinforceflow import logger


class EnvFactory(object):
    @staticmethod
    def make(env, stack_len=None, action_repeat=None, random_start=0, use_smart_wrap=True):
        """Wraps environment into `reinforceflow.envs.EnvWrapper`.

        Args:
            env: String or gym.Wrapper environment subclass.
            stack_len (int): Number of previous frames to stack to the current.
                             Used for the agents without short-term sumtree.
            action_repeat (int): Number of frames to skip with last action repeated.
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
                logger.info('Creating wrapper around Atari environment.')
                return AtariWrapper(env=env,
                                    action_repeat=action_repeat or 4,
                                    stack_len=stack_len or 4)
        logger.info('Creating environment wrapper.')
        return EnvWrapper(env=env, action_repeat=action_repeat)
