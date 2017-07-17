from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import six
from six.moves import range
import gym
from gym import spaces
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

import reinforceflow as rf
from reinforceflow import error


class RawGymWrapper(gym.Wrapper):
    """Light wrapper around `gym.Wrapper` environments.

    Args:
        env: Environment's name string or Gym environment instance.

    Attributes:
        env: Environment instance.
        has_multiple_action (bool): True, if env has multi-discrete action space.
        is_cont_action (bool): True, if env has multi-continuous action space.
    """
    def __init__(self, env):
        if isinstance(env, six.string_types):
            env = gym.make(env)
        super(RawGymWrapper, self).__init__(env)
        self.has_multiple_action = not isinstance(env.action_space, spaces.Discrete)
        self.is_cont_action = self._is_continuous(env.action_space)
        seed = rf.get_random_seed()
        if seed and hasattr(self.env, 'seed'):
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

    def copy(self):
        return copy.deepcopy(self)


class EnvWrapper(RawGymWrapper):
    def __init__(self, env, action_repeat=1):
        # TODO: move frame stack here, for all spaces
        super(EnvWrapper, self).__init__(env)
        self._frame_stack = None
        self._action_repeat = action_repeat or 1

    def _step(self, action):
        reward_total = 0
        for _ in range(self._action_repeat):
            obs, reward, done, info = super(EnvWrapper, self)._step(action)
            reward_total += reward
            if done:
                break
        return obs, reward_total, done, info


class ScreenWrapper(EnvWrapper):
    def __init__(self, env, stack_len=1, action_repeat=1, height=None, width=None, to_gray=True):
        super(ScreenWrapper, self).__init__(env, action_repeat=action_repeat)
        if not isinstance(self.observation_space, spaces.Box):
            raise error.UnsupportedSpace('%s expects Box observation space '
                                         'with pixel screen inputs' % self.class_name())
        if len(self.observation_space.shape) not in [2, 3]:
            raise error.UnsupportedSpace('%s expects Box observation space '
                                         'with pixel screen inputs' % self.class_name())
        self.height = height
        self.width = width
        self.to_gray = to_gray
        self._prev_frame = None
        self._stack_len = stack_len or 1
        self._needs_stack_reset = False
        low = np.min(self.observation_space.low)
        high = np.max(self.observation_space.high)
        self.env.observation_space = spaces.Box(low, high,
                                                (self.height, self.width, self._stack_len))

    def _step(self, action):
        stack_reset = self._needs_stack_reset
        self._needs_stack_reset = False
        reward_total = 0
        for _ in range(self._action_repeat):
            obs, reward, done, info = super(EnvWrapper, self)._step(action)
            reward_total += reward
            if done:
                self._needs_stack_reset = True
                break
        return self._observation(obs, reset=stack_reset), reward_total, done, info

    def _observation(self, obs, reset=False):
        obs = super(ScreenWrapper, self)._observation(obs)
        obs = self._preprocess(obs)
        if not self._stack_len or self._stack_len == 1:
            return obs
        return self._stack_frames(obs, reset)

    def _preprocess(self, obs):
        obs = obs.squeeze()
        if self.to_gray:
            obs = rgb2gray(obs)
        if self.height and self.width:
            obs = resize(obs, (self.height, self.width))
        obs = np.expand_dims(obs, 0)
        if len(obs.shape) <= 3:
            obs = np.expand_dims(obs, 3)
        return obs

    def _stack_frames(self, frame, reset):
        if reset or self._frame_stack is None:
            self._frame_stack = np.repeat(frame, self._stack_len, axis=3)
        else:
            self._frame_stack = np.append(frame, self._frame_stack[:, :, :, :self._stack_len - 1],
                                          axis=3)
        return self._frame_stack


class AtariWrapper(ScreenWrapper):
    def __init__(self, env, stack_len, action_repeat=4, height=84, width=84,
                 to_gray=True, use_merged_frame=True):
        super(AtariWrapper, self).__init__(env=env,
                                           stack_len=stack_len,
                                           action_repeat=action_repeat,
                                           height=height,
                                           width=width,
                                           to_gray=to_gray)
        self.has_lives = hasattr(self.env, 'ale') and hasattr(self.env.ale, 'lives')
        self._use_merged_frame = use_merged_frame
        self._prev_obs = None

    def _step(self, action):
        if self.has_lives:
            start_lives = self.env.ale.lives()
        stack_reset = self._needs_stack_reset
        self._needs_stack_reset = False
        reward_total = 0
        for _ in range(self._action_repeat):
            obs, reward, done, info = RawGymWrapper._step(self, action)
            reward_total += reward
            if self.has_lives and self.env.ale.lives() < start_lives:
                self._needs_stack_reset = True
                break
            if done:
                self._needs_stack_reset = True
                break
        return self._observation(obs, reset=stack_reset), reward_total, done, info

    def _observation(self, obs, reset=False):
        obs = RawGymWrapper._observation(self, obs)
        obs = self._preprocess(obs)
        # Takes maximum value for each pixel value over the current and previous frame.
        # Used to get around Atari sprites flickering (see Mnih et al. (2015)).
        if self._use_merged_frame and not reset:
            prev_obs = self._prev_obs
            self._prev_frame = obs
            obs = np.maximum.reduce([obs, prev_obs]) if prev_obs else obs
        if not self._stack_len or self._stack_len == 1:
            return obs
        return self._stack_frames(obs, reset)
