from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym import spaces
import numpy as np
from reinforceflow import error


class EnvWrapper(object):
    """Light wrapper around `gym.core.Env`.
    Does basic preprocessings to simplify integration with algorithms.

    Args:
        env (gym.core.Env): TODO

    Attributes:
        env (gym.core.Env): TODO
        is_cont_action (bool): TODO
        obs_info (): TODO
        obs_shape (): TODO
        action_shape (): TODO
        action_info (): TODO
    """
    def __init__(self, env):
        self.env = env
        self.has_multiple_action = not isinstance(env.action_space, spaces.Discrete)
        self.is_cont_action = self.is_continuous(env.action_space)
        self.action_shape = self.space_shape(env.action_space)
        self.action_info = self.space_info(env.action_space)
        self.is_cont_obs = self.is_continuous(env.observation_space)
        self.obs_info = self.space_info(env.observation_space)
        self.obs_shape = self.space_shape(env.observation_space)

    def _obs2vec(self, obs):
        if self.is_cont_obs:
            return obs[None, :]

        obs_vectorized = np.zeros(self.obs_shape, dtype='uint8')
        if not isinstance(obs, list):
            obs = list(obs)

        assert len(obs) == len(self.obs_info)
        offset = 0
        for feature, info in zip(obs, self.obs_info):
            obs_vectorized[offset + feature - info['low']] = 1
            offset += info['size']
        return [None] + obs_vectorized

    def prepare_action(self, prediction):
        """
        Converts raw prediction into valid _action for current environment
        Args:
            prediction: Raw output from predictor

        Returns:
            An _action ready for plugging into environment

        Examples:
            >>> preds = model(observation)
            >>> _action = env.prepare_action(preds)
            >>> env.step(_action)
            >>> # ...
        """
        if self.is_cont_action:
            return prediction

        decoded_action = []
        offset = 0
        # Process cases with `Tuple<Discrete>` _action spaces
        for info in self.action_info:
            decoded_action.append(np.argmax(prediction[offset:info['size']]))
            offset += info['size']
        decoded_action = decoded_action[0] if len(decoded_action) == 1 else decoded_action

        assert self.env.action_space.contains(decoded_action)
        return decoded_action

    def step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        return self._obs2vec(obs_next), reward, done, info

    def reset(self):
        return self._obs2vec(self.env.reset())

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

    def seed(self, *args, **kwargs):
        return self.env.seed(*args, **kwargs)

    def configure(self, *args, **kwargs):
        return self.env.configure(*args, **kwargs)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def spec(self):
        return self.env.spec

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __str__(self):
        return 'EnvWrapper(%s)' % self.env

    def __repr__(self):
        return str(self)

    @staticmethod
    def is_continuous(space):
        return isinstance(space, spaces.Box)

    @classmethod
    def space_info(cls, space):
        """TODO
        Args:
            space:

        Returns:
            Tuple of space (high, low) values
        """
        if isinstance(space, spaces.Box):
            return list(space.shape)
        if isinstance(space, spaces.Discrete):
            return [{'low': 0, 'high': space.n, 'size': space.n}]
        if isinstance(space, spaces.MultiDiscrete):
            return [{'low': low, 'high': high, 'size': high - low} for low, high in zip(space.low, space.high)]
        if isinstance(space, spaces.Tuple):
            length = []
            for subspace in space.spaces:
                if isinstance(subspace, spaces.Box):
                    raise error.UnsupportedSpace("Nested Box spaces are not supported yet.")
                length += cls.space_info(subspace)
            return length

    @classmethod
    def space_shape(cls, space):
        """TODO
        Args:
            space:

        Returns:
            Size of the space
        """
        if isinstance(space, spaces.Box):
            return list(space.shape)
        if isinstance(space, spaces.Discrete):
            return space.n
        if isinstance(space, spaces.MultiDiscrete):
            return np.sum(space.high - space.low)
        if isinstance(space, spaces.Tuple):
            length = 0
            for subspace in space.spaces:
                if isinstance(subspace, spaces.Box):
                    raise error.UnsupportedSpace("Nested Box spaces are not supported yet.")
                length += cls.space_info(subspace)
            return length
