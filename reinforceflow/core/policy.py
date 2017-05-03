from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random


class Policy(object):
    def select_action(self, *args, **kwargs):
        raise NotImplementedError


class GreedyPolicy(Policy):
    def __init__(self):
        self.__apply__ = self.select_action

    def select_action(self, prediction, env):
        return env.prepare_action(prediction)


class EGreedyPolicy(Policy):
    def __init__(self, eps_start, eps_final, anneal_steps):
        self._start = eps_start
        self._final = eps_final
        self._anneal_steps = anneal_steps
        self._epsilon = self._start
        self._anneal_range = self._start - self._final
        self.epsilon = eps_start

    def select_action(self, prediction, env, step):
        self.epsilon = self._update_epsilon(step)
        if random.random() > self.epsilon:
            return env.prepare_action(prediction)
        else:
            return env.action_space.sample()

    def _update_epsilon(self, step):
        if step >= self._anneal_steps:
            return self._final
        return self._start - (step / self._anneal_steps) * self._anneal_range
