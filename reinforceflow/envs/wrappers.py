from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym import spaces
from skimage.transform import resize
from skimage.color import rgb2gray
from reinforceflow.envs.env_wrapper import EnvWrapper
from reinforceflow import error


def ScreenWrapper(stack_len, height, width, action_repeat=1, to_gray=True):
    class ScreenWrapper(EnvWrapper):
        def __init__(self, env):
            super(ScreenWrapper, self).__init__(env)
            if not isinstance(self.observation_space, spaces.Box):
                raise error.UnsupportedSpace('%s expects Box observation space with pixel screen inputs'
                                             % self.class_name())
            if len(self.observation_space.shape) not in [2, 3]:
                raise error.UnsupportedSpace('%s expects Box observation space with pixel screen inputs'
                                             % self.class_name())

            self._frame_stack = None
            self._prev_frame = None
            self.height = height
            self.width = width
            self.to_gray = to_gray
            self.stack_len = stack_len or 1
            self._action_repeat = action_repeat or 1

            low = np.min(self.observation_space.low)
            high = np.max(self.observation_space.high)
            self.env.observation_space = spaces.Box(low, high, (self.height, self.width, self.stack_len))

        def _step(self, action):
            stack_reset = self._needs_stack_reset
            self._needs_stack_reset = False
            reward_total = 0
            for _ in range(self._action_repeat):
                obs, reward, done, info = super(ScreenWrapper, self)._step(action)
                reward_total += reward
                if done:
                    self._needs_stack_reset = True
                    break
            return self._observation(obs, reset=stack_reset), reward_total, done, info

        def _observation(self, obs, reset=False):
            obs = super(ScreenWrapper, self)._observation(obs)
            obs = self._preprocess(obs)
            if not self.stack_len or self.stack_len == 1:
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
                self._frame_stack = np.repeat(frame, self.stack_len, axis=3)
            else:
                self._frame_stack = np.append(frame, self._frame_stack[:, :, :, :self.stack_len - 1], axis=3)
            return self._frame_stack

    return ScreenWrapper


def AtariWrapper(stack_len, height, width, action_repeat=1, to_gray=True, use_merged_frame=True):
    class AtariWrapper(ScreenWrapper(stack_len, height, width, action_repeat, to_gray)):
        def __init__(self, env):
            super(AtariWrapper, self).__init__(env)
            self.has_lives = hasattr(self.env, 'ale') and hasattr(self.env.ale, 'lives')
            self._use_merged_frame = use_merged_frame
            self._needs_stack_reset = False
            self._prev_obs = None

        def _step(self, action):
            if self.has_lives:
                start_lives = self.env.ale.lives()
            stack_reset = self._needs_stack_reset
            self._needs_stack_reset = False
            reward_total = 0
            for _ in range(self._action_repeat):
                obs, reward, done, info = EnvWrapper._step(self, action)
                reward_total += reward
                if self.has_lives and self.env.ale.lives() < start_lives:
                    self._needs_stack_reset = True
                    break
                if done:
                    self._needs_stack_reset = True
                    break
            return self._observation(obs, reset=stack_reset), reward_total, done, info

        def _observation(self, obs, reset=False):
            obs = EnvWrapper._observation(self, obs)
            obs = self._preprocess(obs)
            # Takes maximum value for each pixel value over the current and previous frame.
            # Used to get around Atari sprites flickering (see Mnih et al. (2015)).
            if self._use_merged_frame and not reset:
                prev_obs = self._prev_obs
                self._prev_frame = obs
                obs = np.maximum.reduce([obs, prev_obs]) if prev_obs else obs
            if not self.stack_len or self.stack_len == 1:
                return obs
            return self._stack_frames(obs, reset)

    return AtariWrapper
