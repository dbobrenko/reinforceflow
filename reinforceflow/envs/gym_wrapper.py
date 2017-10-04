from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import numpy as np
import gym
from gym import spaces
import reinforceflow
from reinforceflow.envs.env_wrapper import Env
from reinforceflow.core.spaces import DiscreteOneHot, Tuple, Continious
from reinforceflow.utils import stack_observations, image_preprocess, one_hot


def to_rf_space(space):
    """Converts Gym space instance into Reinforceflow's."""
    if isinstance(space, spaces.Discrete):
        return DiscreteOneHot(space.n)

    if isinstance(space, spaces.MultiDiscrete):
        # space.low > 0 will lead to unused first n actions.
        # return Tuple([DiscreteOneHot(n) for n in space.high])
        raise ValueError("MultiDiscrete spaces aren't supported yet.")

    if isinstance(space, spaces.MultiBinary):
        return Tuple([DiscreteOneHot(2) for _ in space.n])

    if isinstance(space, spaces.Box):
        return Continious(space.low, space.high)

    if isinstance(space, spaces.Tuple):
        converted_spaces = []
        for sub_space in space.spaces:
            converted_spaces.append(to_rf_space(sub_space))
        return Tuple(*converted_spaces)
    raise ValueError("Unsupported space %s." % space)


def make_gym2rf_converter(space):
    """Makes space converter function that maps space samples Gym -> rf."""
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
            sub_converters.append(make_gym2rf_converter(sub_space))

        def converter(sample):
            converted_tuple = []
            for sub_sample, sub_converter in zip(sample, sub_converters):
                converted_tuple.append(sub_converter(sub_sample))
            return tuple(converted_tuple)
        return converter
    raise ValueError("Unsupported space %s." % space)


def make_rf2gym_converter(space):
    """Makes space converter function that maps space samples rf -> Gym."""
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
            sub_converters.append(make_rf2gym_converter(sub_space))

        def converter(sample):
            converted_tuple = []
            for sub_sample, sub_converter in zip(sample, sub_converters):
                converted_tuple.append(sub_converter(sub_sample))
            return tuple(converted_tuple)
        return converter
    raise ValueError("Unsupported space %s." % space)


class GymWrapper(Env):
    """Light wrapper around OpenAI Gym and Universe environments.
    See `Env`.
    """
    def __init__(self, env, action_repeat=1, obs_stack=1):
        if isinstance(env, six.string_types):
            env = gym.make(env)
        if isinstance(env.action_space, spaces.MultiDiscrete):
            raise ValueError("Gym environments with MultiDiscrete spaces aren't supported yet.")

        super(GymWrapper, self).__init__(env,
                                         obs_space=to_rf_space(env.observation_space),
                                         action_space=to_rf_space(env.action_space),
                                         action_repeat=action_repeat,
                                         obs_stack=obs_stack)
        self._obs_to_rf = make_gym2rf_converter(self.obs_space)
        self._action_to_rf = make_rf2gym_converter(self.action_space)
        self._action_to_gym = make_rf2gym_converter(self.action_space)
        seed = reinforceflow.get_random_seed()
        if seed and hasattr(self.env, 'seed'):
            self.env.seed(seed)

    def _step(self, action):
        gym_action = self._action_to_gym(action)
        obs, reward, done, info = self.env.step(gym_action)
        return self._obs_to_rf(obs), reward, done, info

    def _reset(self):
        return self._obs_to_rf(self.env.reset())

    def render(self):
        self.env.render()


class GymPixelWrapper(GymWrapper):
    def __init__(self,
                 env,
                 action_repeat,
                 obs_stack,
                 to_gray=False,
                 resize_width=None,
                 resize_height=None,
                 merge_last_frames=False):
        super(GymPixelWrapper, self).__init__(env,
                                              action_repeat=action_repeat,
                                              obs_stack=obs_stack)
        if not isinstance(self.obs_space, Continious) or len(self.obs_space.shape) != 3:
            raise ValueError('%s expects observation space with pixel inputs; '
                             'i.e. 3-D tensor (H, W, C).' % self.__class__.__name__)
        # if self.obs_space.shape[-1] not in [1, 3]:
        #     raise ValueError('%s expects input observations '
        #                      'with channel size equal to 1 or 3.' % self.__class__.__name__)
        self._height = resize_height
        self._width = resize_width
        self._to_gray = to_gray
        new_shape = list(self.obs_space.shape)
        new_shape[0] = resize_height if resize_height else new_shape[0]
        new_shape[1] = resize_width if resize_width else new_shape[1]
        if to_gray:
            new_shape[-1] = obs_stack
        self.obs_space.reshape(tuple(new_shape))

        self._use_merged_frame = merge_last_frames
        self.has_lives = hasattr(self.env, 'ale') and hasattr(self.env.ale, 'lives')
        self._prev_obs = None

    def step(self, action):
        """See `Env.step`."""
        start_lives = self.env.ale.lives() if self.has_lives else 0
        reward_total = 0
        done = False
        needs_stack_reset = False
        for _ in range(self._action_repeat):
            obs, reward, done, info = self._step(action)
            reward_total += reward
            if done or self.has_lives and self.env.ale.lives() < start_lives:
                needs_stack_reset = True
                break
        obs = self._obs_preprocess(obs)
        # Observation stacking
        if self._obs_stack_len > 1:
            obs = stack_observations(obs, self._obs_stack_len, self._obs_stack)
            # Reset observations stack whenever last step is terminal
            if needs_stack_reset:
                self._obs_stack = None
                self._prev_obs = None
        return obs, reward_total, done, info

    def _reset(self):
        return self._obs_to_rf(self._obs_preprocess(self.env.reset()))

    def _obs_preprocess(self, obs):
        """Applies such image preprocessing as resizing and converting to grayscale.
        Also, takes maximum value for each pixel value over the current and previous frame.
        Used to get around Atari sprites flickering (see Mnih et al. (2015)).

        Args:
            obs (numpy.ndarray): 2-D or 3-D observation.
        Returns:
            (numpy.ndarray) Preprocessed 3-D observation.
        """
        obs = image_preprocess(obs, resize_height=self.height,
                               resize_width=self.width, to_gray=self._to_gray)
        if self._use_merged_frame and self._prev_obs is not None:
            prev_obs = self._prev_obs
            self._prev_obs = obs
            obs = np.maximum.reduce([obs, prev_obs]) if prev_obs else obs
        return obs

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def is_grayscale(self):
        return self._to_gray
