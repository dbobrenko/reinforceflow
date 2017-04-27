from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import numpy as np
from reinforceflow import error
from reinforceflow.envs.env_wrapper import EnvWrapper


class BaseAgent(object):
    def fit(self):
        pass

    def train_on_batch(self, states, actions, rewards):
        pass


class DiscreteAgent(BaseAgent):
    """Base class for Agent with discrete _action space.
    Args:
        env (reinforceflow.EnvWrapper): Environment Wrapper
        epsilon (float): The probability for epsilon-greedy exploration, expected to be in range [0; 1]
        gamma (float): Discount factor
    """
    # TODO: Check for env type, raise errors
    def __init__(self, env, gamma, epsilon):
        super(DiscreteAgent, self).__init__()
        if not isinstance(env, EnvWrapper):
            env = EnvWrapper(env)
        if env.is_cont_action:
            raise error.UnsupportedSpace('%s does not support environments with continuous _action space.'
                                         % self.__class__.__name__)
        if env.has_multiple_action:
            raise error.UnsupportedSpace('%s does not support environments with multiple _action spaces.'
                                         % self.__class__.__name__)
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma

    def train_on_batch(self, states, actions, rewards):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError


class TableAgent(DiscreteAgent):
    """Base class for Table-based Agent with discrete observation and _action space.
    Args:
        env (reinforceflow.EnvWrapper): Environment Wrapper
        epsilon (float): The probability for epsilon-greedy exploration, expected to be in range [0; 1]
        gamma (float): Discount factor
        lr (float): Learning rate
    """
    def __init__(self, *args, **kwargs):
        super(TableAgent, self).__init__(*args, **kwargs)
        if self.env.is_cont_obs:
            raise error.UnsupportedSpace('%s does not support environments with continuous observation space.'
                                         % self.__class__.__name__)
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_shape))

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def train_on_batch(self, states, actions, rewards):
        raise NotImplementedError
