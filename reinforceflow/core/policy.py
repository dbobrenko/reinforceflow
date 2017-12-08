from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np

from reinforceflow.utils import one_hot


class BasePolicy(object):
    def select_action(self, *args, **kwargs):
        raise NotImplementedError


class GreedyPolicy(BasePolicy):
    @classmethod
    def select_action(cls, env, prediction, *args, **kwargs):
        return one_hot(env.action_space.shape, np.argmax(prediction))


class EGreedyPolicy(BasePolicy):
    def __init__(self, eps_start, eps_final, anneal_steps):
        self._start = eps_start
        self._final = eps_final
        self._anneal_steps = anneal_steps
        self._epsilon = self._start
        self._anneal_range = self._start - self._final
        self.epsilon = eps_start

    def select_action(self, env, prediction, step):
        self.epsilon = self._update_epsilon(step)
        if random.random() > self.epsilon:
            return one_hot(env.action_space.shape, np.argmax(prediction))
        else:
            return env.action_space.sample()

    def _update_epsilon(self, step):
        if step >= self._anneal_steps:
            return self._final
        return self._start - (step / self._anneal_steps) * self._anneal_range
