from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from reinforceflow.envs.env_wrapper import EnvWrapper
from reinforceflow.envs.ale_wrapper import ALEWrapper

from reinforceflow.core import NullPreprocessor, AtariPreprocessor


class EnvFactory(object):
    @staticmethod
    def make(env, preprocessor=None, action_repeat=4, random_start=0):
        if isinstance(env, str):
            env = gym.make(env)
        if hasattr(env, 'ale'):
            return ALEWrapper(env,
                              preprocessor=preprocessor or AtariPreprocessor(stack_len=4),
                              action_repeat=action_repeat,
                              random_start=random_start)
        else:
            return EnvWrapper(env,
                              preprocessor=preprocessor or NullPreprocessor(),
                              action_repeat=action_repeat,
                              random_start=random_start)
