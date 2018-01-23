from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import six

from reinforceflow.envs import copyable, ObservationStackWrap, ActionRepeatWrap, \
    Vectorize, RandomNoOpWrap, RewardClipWrap, ImageWrap, NormalizeImageWrap


@copyable
class FireResetWrap(gym.Wrapper):
    def __init__(self, env, fire_action):
        super(FireResetWrap, self).__init__(env=env)
        assert self.action_space.contains(fire_action),\
            "Invalid action %s for %s environment." % (fire_action, self.env)
        self._start_action = fire_action

    def _reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(self._start_action)
        if done:
            obs = self.env.reset(**kwargs)
        return obs


@copyable
class AtariLifeReset(FireResetWrap):
    def __init__(self, env, fire_action):
        super(AtariLifeReset, self).__init__(env=env, fire_action=fire_action)
        self.env = FireResetWrap(self.env, fire_action=fire_action)
        assert hasattr(self.env.unwrapped, 'ale')
        self._lives = 0
        self._needs_reset = True

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._needs_reset = done
        lives = self.env.unwrapped.ale.lives()
        if self._lives > lives > 0:
            if not done:
                info['live_lost'] = True
            done = True
        self._lives = lives
        return obs, reward, done, info

    def _reset(self, **kwargs):
        if self._needs_reset:
            obs = self.env.reset()
        else:
            obs, _, _, _ = self.env.step(self._start_action)
        self._lives = self.env.unwrapped.ale.lives()
        return obs


@copyable
class AtariObsStackLifeReset(FireResetWrap):
    def __init__(self, env, obs_stack, fire_action=None):
        super(AtariObsStackLifeReset, self).__init__(env=env, fire_action=fire_action)
        if fire_action is not None:
            self.env = AtariLifeReset(self.env, fire_action=fire_action)
        self.env = ObservationStackWrap(self.env, obs_stack)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._needs_reset = done
        lives = self.env.unwrapped.ale.lives()
        if self._lives > lives > 0 and not done:
            # TODO: Check if after live lost obs is new or old (expected to be the new one).
            self.env.reset_stack()
        self._lives = lives
        return obs, reward, done, info

    def _reset(self, **kwargs):
        if self._needs_reset:
            obs = self.env.reset()
        else:
            obs, _, _, _ = self.env.step(self._start_action)
        self._lives = self.env.unwrapped.ale.lives()
        return obs


@copyable
class ActionRepeatMaxWrap(ActionRepeatWrap):
    """Repeats last action given number of times and applies max pooling to the last two frames.
    Args:
        action_repeat (int): The number of step on which the action will be repeated.
    """
    def __init__(self, env, action_repeat):
        super(ActionRepeatMaxWrap, self).__init__(env=env, action_repeat=action_repeat)

    def _step(self, action):
        obs, reward_total, done, info_all = self.env.step(action)
        last_obses = [obs, obs]
        for _ in range(self._action_repeat - 1):
            obs, reward, done, info = self.env.step(action)
            reward_total += reward
            info_all.update(info)
            last_obses[0] = last_obses[1]
            last_obses[1] = obs
            if done:
                break
        # Takes maximum value for each pixel value over the current and previous frame.
        # Used to get around Atari sprites flickering (Mnih et al. (2015))
        if self._action_repeat > 1:
            obs = np.maximum.reduce(last_obses)
        return obs, reward_total, done, info_all


@copyable
class AtariWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 fire_action=None,
                 noop_action=None,
                 action_repeat=4,
                 obs_stack=4,
                 to_gray=True,
                 new_width=84,
                 new_height=84,
                 clip_rewards=True):
        if isinstance(env, six.string_types):
            env = gym.make(env)
        super(AtariWrapper, self).__init__(env=env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.env = Vectorize(self.env)
        if noop_action:
            self.env = RandomNoOpWrap(self.env, noop_action=noop_action)
        self.env = ActionRepeatMaxWrap(self.env, action_repeat=action_repeat)
        if fire_action:
            self.env = AtariLifeReset(self.env, fire_action=fire_action)
        if clip_rewards:
            self.env = RewardClipWrap(self.env)
        self.env = ImageWrap(self.env, to_gray=to_gray, new_width=new_width, new_height=new_height)
        self.env = NormalizeImageWrap(self.env)
        self.env = ObservationStackWrap(self.env, obs_stack=obs_stack)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
