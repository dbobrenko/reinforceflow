from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import numpy as np
from reinforceflow import error
from reinforceflow.envs.env_wrapper import EnvWrapper


class BaseAgent(object):
    def __init__(self, env):
        if not isinstance(env, EnvWrapper):
            env = EnvWrapper(env)
        self.env = env

    def train(self, *args, **kwargs):
        pass


class DiscreteAgent(BaseAgent):
    """Base class for Agent with discrete action space"""
    def __init__(self, env):
        super(DiscreteAgent, self).__init__(env)
        if self.env.is_cont_action:
            raise error.UnsupportedSpace('%s does not support environments with continuous action space.'
                                         % self.__class__.__name__)
        if self.env.has_multiple_action:
            raise error.UnsupportedSpace('%s does not support environments with multiple action spaces.'
                                         % self.__class__.__name__)
        self.env = env


class TableAgent(DiscreteAgent):
    """Base class for Table-based Agent with discrete observation and action space"""
    def __init__(self, env):
        super(TableAgent, self).__init__(env)
        if self.env.is_cont_obs:
            raise error.UnsupportedSpace('%s does not support environments with continuous observation space.'
                                         % self.__class__.__name__)
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_shape))
