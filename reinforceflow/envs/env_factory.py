from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import gym
from reinforceflow.envs.env_wrapper import EnvWrapper
from reinforceflow.envs.wrappers import AtariWrapper
from reinforceflow import logger


class EnvFactory(object):
    @staticmethod
    def make(env, action_repeat=4, random_start=0, enable_smart_wrap=True):
        """Wraps environment into GymWrapper.

        Args:
            env: string or gym.Wrapper environment subclass.
            action_repeat (int): number of frames to skip with last action repeated.
            random_start (int): (TODO) determines the max number of randomly skipped frames,
                                see Mnih et al., 2015.
            enable_smart_wrap (bool): Enable smart wrapping. E.g.:
                              Atari environments will be processed as stated in Mnih et al., 2015.

        Returns (gym.Wrapper): environment instance.
        """
        # TODO: add random_start functionality
        if enable_smart_wrap:
            if hasattr(env, 'env') and hasattr(env.env, 'ale'):
                logger.info('Creating wrapper around Atari environment.')
                return AtariWrapper(stack_len=4,
                                    height=84,
                                    width=84,
                                    action_repeat=action_repeat,
                                    to_gray=True,
                                    use_merged_frame=True)(env)
        logger.info('Creating environment wrapper.')
        return EnvWrapper(env)


def make_new_env(env):
    if isinstance(env, six.string_types):
        return EnvFactory.make(env)
    elif isinstance(env, gym.Wrapper):
        return EnvFactory.make(env.spec.id)
    else:
        raise ValueError('Unknown environment type. '
                         'Expected to be a string or an instance of gym.Wrapper.')
