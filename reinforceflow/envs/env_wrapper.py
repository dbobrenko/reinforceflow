from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym import spaces
import numpy as np
import reinforceflow as rf
from reinforceflow import error


class EnvWrapper(gym.Wrapper):
    """Light wrapper around `gym.core.Env`.
    Does basic preprocessings to simplify integration with algorithms.

    Args:
        env (gym.core.Env): TODO

    Attributes:
        env (gym.core.Env): TODO
        is_cont_action (bool): TODO
    """
    def __init__(self, env):
        super(EnvWrapper, self).__init__(env)
        self.has_multiple_action = not isinstance(env.action_space, spaces.Discrete)
        self.is_cont_action = self._is_continuous(env.action_space)
        # obs_space = self.env.observation_space
        # if isinstance(obs_space, spaces.Box):
        #     self._observation = self._obs_box
        # elif isinstance(obs_space, spaces.Discrete):
        #     self._observation = self._obs_discrete
        # elif isinstance(obs_space, spaces.Tuple):
        #     self._observation = self._obs_tuple
        # else:
        #     raise error.UnsupportedSpace('Unsupported space %s' % obs_space).contains(action):
        # action_space = self.env.observation_space
        # if isinstance(action_space, spaces.Box):
        #     self._observation = self._action_box
        # if isinstance(action_space, spaces.Discrete):
        #     self._action = self._action_discrete
        # elif isinstance(action_space, spaces.Tuple):
        #     self._action = self._action_tuple
        # else:
        #     raise error.UnsupportedSpace('Unsupported space %s' % obs_space).contains(action):
        seed = rf.get_random_seed()
        if seed:
            self.env.seed(seed)

    def _observation(self, obs):
        obs_space = self.env.observation_space

        if isinstance(obs_space, spaces.Box):
            return np.expand_dims(obs, 0)

        elif isinstance(obs_space, spaces.Discrete):
            return np.expand_dims(self._one_hot(obs_space.n, obs), 0)

        elif isinstance(obs_space, spaces.Tuple):
            shape = []
            for subspace in obs_space.spaces:
                if isinstance(subspace, spaces.Tuple):
                    raise error.UnsupportedSpace("Nested Tuple spaces are not supported")
                shape.append(self._observation(subspace))
            return shape
        else:
            raise error.UnsupportedSpace('Unsupported space %s' % obs_space)

    def prepare_action(self, action):
        return self._action(action)

    def _action(self, action):
        if self.env.action_space.contains(action):
            return action

        action_space = self.env.action_space
        if isinstance(action_space, spaces.Discrete):
            return np.argmax(action)

        elif isinstance(action_space, spaces.Tuple):
            shape = []
            for subspace in action_space.spaces:
                if isinstance(subspace, spaces.Tuple):
                    raise error.UnsupportedSpace("Nested Tuple spaces are not supported")
                shape.append(self._action(subspace))
            return shape
        else:
            raise error.UnsupportedSpace('Unsupported space %s' % action_space)

    def _step(self, action):
        obs, reward, done, info = self.env.step(self._action(action))
        return self._observation(obs), reward, done, info

    def _reset(self):
        return self._observation(self.env.reset())

    @property
    def observation_shape(self):
        return self._space_shape(self.env.observation_space)

    @property
    def action_shape(self):
        return self._space_shape(self.env.action_space)

    @classmethod
    def _one_hot(cls, shape, idx):
        vec = np.zeros(shape, dtype=np.uint8)
        vec[idx] = 1
        return vec

    @staticmethod
    def _is_continuous(space):
        return isinstance(space, spaces.Box)

    @classmethod
    def _space_shape(cls, space):
        if isinstance(space, spaces.Box):
            return list(space.shape)
        if isinstance(space, spaces.Discrete):
            return space.n
        if isinstance(space, spaces.MultiDiscrete):
            raise error.UnsupportedSpace("MultiDiscrete spaces are not supported yet.")
        if isinstance(space, spaces.Tuple):
            shape = []
            for subspace in space.spaces:
                if isinstance(space, spaces.Tuple):
                    raise error.UnsupportedSpace("Nested Tuple spaces are not supported")
                shape.append(cls._space_shape(subspace))
            return shape
