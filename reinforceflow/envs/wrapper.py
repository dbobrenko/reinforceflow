from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import gym
import numpy as np
import six
from gym import spaces

import reinforceflow as rf
from reinforceflow.core.space import DiscreteOneHot, Tuple, Continuous
from reinforceflow.utils import image_preprocess, one_hot


def copyable(cls):
    """Decorator, that makes possible to copy the env with it's starting arguments.
    A temporary workaround, since copy.deepcopy works poorly on some environments."""
    init = cls.__init__

    def __init__(self, env, *args, **kwargs):
        init(self, env, *args, **kwargs)
        if isinstance(env, str):
            self.copy = lambda *a, **kw: cls(gym.make(env), *args, **kwargs)
        else:
            self.copy = lambda *a, **kw: cls(copy.deepcopy(env), *args, **kwargs)
        self.__copy__ = self.copy
        self.__deepcopy__ = self.copy

    cls.__init__ = __init__
    return cls


@copyable
class Vectorize(gym.Wrapper):
    """Vectorizes OpenAI Gym spaces."""
    def __init__(self, env):
        if isinstance(env, six.string_types):
            env = gym.make(env)
        super(Vectorize, self).__init__(env)
        if isinstance(env.action_space, spaces.MultiDiscrete):
            raise ValueError("Gym environments with MultiDiscrete spaces aren't supported yet.")
        self.observation_space = self.vectorize_space(self.env.observation_space)
        self.action_space = self.vectorize_space(self.env.action_space)
        self._obs_to_rf = self.make_gym2vec_converter(self.observation_space)
        self._action_to_rf = self.make_gym2vec_converter(self.action_space)
        self._action_to_gym = self.make_vec2gym_converter(self.action_space)
        seed = rf.get_random_seed()
        if seed and hasattr(self.env, 'seed'):
            self.env.seed(seed)

    def _step(self, action):
        gym_action = self._action_to_gym(action)
        obs, reward, done, info = self.env.step(gym_action)
        return self._obs_to_rf(obs), reward, done, info

    def _reset(self):
        obs = self._obs_to_rf(self.env.reset())
        return obs

    @staticmethod
    def vectorize_space(space):
        """Converts Gym space into vectorized space."""
        if isinstance(space, spaces.Discrete):
            return DiscreteOneHot(space.n)

        if isinstance(space, spaces.MultiDiscrete):
            # space.low > 0 will lead to unused first n actions.
            # return Tuple([DiscreteOneHot(n) for n in space.high])
            raise ValueError("MultiDiscrete spaces aren't supported yet.")

        if isinstance(space, spaces.MultiBinary):
            return Tuple([DiscreteOneHot(2) for _ in space.n])

        if isinstance(space, spaces.Box):
            return Continuous(space.low, space.high)

        if isinstance(space, spaces.Tuple):
            converted_spaces = []
            for sub_space in space.spaces:
                converted_spaces.append(Vectorize.vectorize_space(sub_space))
            return Tuple(*converted_spaces)
        raise ValueError("Unsupported space %s." % space)

    @staticmethod
    def make_gym2vec_converter(space):
        """Makes converter function that maps space samples Gym -> Vectorized."""
        # TODO: add spaces.MultiDiscrete support.
        if isinstance(space, spaces.Discrete):
            def converter(sample):
                return one_hot(space.n, sample)
            return converter

        if isinstance(space, spaces.MultiBinary):
            def converter(sample):
                return tuple([one_hot(2, s) for s in sample])
            return converter

        if isinstance(space, spaces.Box):
            return lambda sample: sample

        if isinstance(space, spaces.Tuple):
            sub_converters = []
            for sub_space in space.spaces:
                sub_converters.append(Vectorize.make_gym2vec_converter(sub_space))

            def converter(sample):
                converted_tuple = []
                for sub_sample, sub_converter in zip(sample, sub_converters):
                    converted_tuple.append(sub_converter(sub_sample))
                return tuple(converted_tuple)

            return converter
        raise ValueError("Unsupported space %s." % space)

    @staticmethod
    def make_vec2gym_converter(space):
        """Makes space converter function that maps space samples Vectorized -> Gym."""
        # TODO: add spaces.MultiDiscrete support.
        if isinstance(space, spaces.Discrete):
            def converter(sample):
                return np.argmax(sample)

            return converter

        if isinstance(space, spaces.MultiBinary):
            def converter(sample):
                return tuple([np.argmax(s) for s in sample])

            return converter

        if isinstance(space, spaces.Box):
            return lambda sample: sample

        if isinstance(space, spaces.Tuple):
            sub_converters = []
            for sub_space in space.spaces:
                sub_converters.append(Vectorize.make_vec2gym_converter(sub_space))

            def converter(sample):
                converted_tuple = []
                for sub_sample, sub_converter in zip(sample, sub_converters):
                    converted_tuple.append(sub_converter(sub_sample))
                return tuple(converted_tuple)

            return converter
        raise ValueError("Unsupported space %s." % space)


@copyable
class ImageWrap(gym.Wrapper):
    def __init__(self, env, to_gray=False, new_width=None, new_height=None):
        super(ImageWrap, self).__init__(env=env)
        self._height = new_height
        self._width = new_width
        self._to_gray = to_gray
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        new_shape = list(self.observation_space.shape)
        assert isinstance(self.observation_space, Continuous) and 2 <= len(new_shape) <= 3,\
            "Observation space must be continuous 2-D or 3-D tensor."
        new_shape[0] = new_height if new_height else new_shape[0]
        new_shape[1] = new_width if new_width else new_shape[1]
        # Always add channel dimension
        if len(new_shape) == 2:
            new_shape.append(1)

        # Check for grayscale
        if to_gray:
            new_shape[-1] = 1

        self.observation_space.reshape(tuple(new_shape))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._obs_preprocess(obs), reward, done, info

    def _reset(self):
        return self._obs_preprocess(self.env.reset())

    def _obs_preprocess(self, obs):
        """Applies such image preprocessing as resizing and converting to grayscale.
        Also, takes maximum value for each pixel value over the current and previous frame.
        Used to get around Atari sprites flickering (see Mnih et al. (2015)).

        Args:
            obs (numpy.ndarray): 2-D or 3-D observation.
        Returns:
            (numpy.ndarray) Preprocessed 3-D observation.
        """
        obs = image_preprocess(obs, resize_height=self._height, resize_width=self._width,
                               to_gray=self._to_gray)
        return obs

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def grayscale(self):
        return self._to_gray


@copyable
class ActionRepeatWrap(gym.Wrapper):
    """Repeats last action given number of times.
    Args:
        action_repeat (int): The number of step on which the action will be repeated.
    """
    def __init__(self, env, action_repeat):
        super(ActionRepeatWrap, self).__init__(env=env)
        assert action_repeat > 0, "Action repeat number must be higher than 0."
        self._action_repeat = action_repeat

    def _step(self, action):
        obs, reward_total, done, info_all = self.env.step(action)
        for _ in range(self._action_repeat - 1):
            obs, reward, done, info = self.env.step(action)
            reward_total += reward
            info_all.update(info)
            if done:
                break
        return obs, reward_total, done, info_all


@copyable
class ObservationStackWrap(gym.Wrapper):
    """
    Args:
        obs_stack (int): The length of stacked observations.
            Provided observation_space shape will be automatically modified.
            Doesn't support Tuple spaces.
    """
    def __init__(self, env,  obs_stack):
        super(ObservationStackWrap, self).__init__(env=env)
        assert obs_stack > 1, "Observation stack length must be higher than 1."
        assert not isinstance(self.observation_space, Tuple),\
            "Observation stack is not compatible with Tuple spaces."
        self.stack_len = obs_stack or 1
        self.observation_space = self.env.observation_space
        new_shape = list(self.observation_space.shape)
        new_shape[-1] = self.observation_space.shape[-1] * obs_stack
        self.observation_space.reshape(tuple(new_shape))
        self._obs_stack = None
        self._last_obs = None

    def _reset(self):
        self._last_obs = self.env.reset()
        self.reset_stack()
        return self._obs_stack

    def _step(self, action):
        self._last_obs, reward, done, info = self.env.step(action)
        self._obs_stack = self.stack_observations(self._last_obs, self.stack_len, self._obs_stack)
        return self._obs_stack, reward, done, info

    def reset_stack(self):
        self._obs_stack = self.stack_observations(self._last_obs, self.stack_len)

    @staticmethod
    def stack_observations(obs, stack_len, obs_stack=None):
        """Stacks observations along last axis.
           New observations are appended to the existing stack,
           so the chronological order of observations looks like:
           [Obs{N}, Obs{N-1}, ..., Obs{2}, Obs{1: most recent}]

        Args:
            obs (numpy.ndarray): Observation.
            stack_len (int): Stack's total length.
            obs_stack (numpy.ndarray): Current stack of observations.
                If None, passed `obs` will be repeated for `stack_len` times.

        Returns (numpy.ndarray):
            Stacked observations along last axis.
        """
        stack_axis = np.ndim(obs) - 1
        obs_axis_len = np.shape(obs)[stack_axis]
        if obs_stack is None:
            obs_stack = obs
            # np.repeat won't work correctly, since it repeats each element separately,
            # instead of repeating each observation.
            for i in range(stack_len - 1):
                obs_stack = np.append(obs_stack, obs, axis=stack_axis)
        else:
            # Delete the oldest observation.
            # Note, that a single observation may have several depth channels e.g RGB,
            # so that we need to delete each of its channels separately.
            del_indexes = list(range(0, obs_axis_len))
            obs_previous = np.delete(obs_stack, del_indexes, axis=stack_axis)
            obs_stack = np.append(obs_previous, obs, axis=stack_axis)
        assert obs_stack.shape[stack_axis] // obs_axis_len == stack_len
        return obs_stack


@copyable
class RandomNoOpWrap(gym.Wrapper):
    def __init__(self, env, noop_action, noop_max=30, noop_min=0):
        super(RandomNoOpWrap, self).__init__(env=env)
        assert self.action_space.contains(noop_action),\
            "Invalid action %s for %s environment." % (noop_action, self.env)
        assert noop_max > 0
        assert noop_min >= 0
        self._noop_action = noop_action
        self._noop_max = noop_max
        self._noop_min = noop_min

    def _reset(self):
        self.env.reset()
        skip = np.random.randint(self._noop_min, self._noop_max)
        for _ in range(skip-1):
            # TODO: Skip all wrappers to remove unnecessary preprocessing.
            # obs, _, done, _ = self.unwrapped.step(self._noop_action)
            obs, _, done, _ = self.env.step(self._noop_action)
            if done:
                self.env.reset()
        # Always perform last step with all wrappers applied.
        obs, _, done, _ = self.env.step(self._noop_action)
        return obs


@copyable
class NormalizeImageWrap(gym.ObservationWrapper):
    def _observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


@copyable
class RewardClipWrap(gym.Wrapper):
    def _step(self, action):
        """Clips reward into {-1, 0, 1} range, as suggested in Mnih et al., 2013."""
        obs, reward, done, info = self.env.step(action)
        info['reward_unclip'] = reward
        return obs, np.sign(reward), done, info
