from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import six
import numpy as np
import gym
from gym import spaces
import reinforceflow
from reinforceflow.core.space import DiscreteOneHot, Tuple, Continious
from reinforceflow.utils import stack_observations, image_preprocess, one_hot


def renewable(cls):
    """Decorator, that makes possible to clone the env with it's starting arguments.
    A temporary workaround, since copy.deepcopy works poorly on some environments.
    Adds `new` method, that creates new env from instance."""
    init = cls.__init__

    def __init__(self, env, *args, **kwargs):
        init(self, env, *args, **kwargs)
        if hasattr(env, 'new'):
            self.new = lambda: cls(env.new(), *args, **kwargs)
        elif isinstance(env, str):
            self.new = lambda: cls(gym.make(env), *args, **kwargs)
        else:
            try:
                env_name = env.spec.id
                self.new = lambda: cls(gym.make(env_name), *args, **kwargs)
            except AttributeError:
                self.new = lambda: cls(copy.deepcopy(env), *args, **kwargs)

    cls.__init__ = __init__
    return cls


@renewable
class GymWrapper(gym.Wrapper):
    """Light wrapper around OpenAI Gym and Universe environments."""
    def __init__(self, env):
        if isinstance(env, six.string_types):
            env = gym.make(env)
        super(GymWrapper, self).__init__(env)
        if isinstance(env.action_space, spaces.MultiDiscrete):
            raise ValueError("Gym environments with MultiDiscrete spaces aren't supported yet.")
        self.observation_space = _to_rf_space(self.env.observation_space)
        self.action_space = _to_rf_space(self.env.action_space)
        self._obs_to_rf = _make_gym2rf_converter(self.observation_space)
        self._action_to_rf = _make_gym2rf_converter(self.action_space)
        self._action_to_gym = _make_rf2gym_converter(self.action_space)
        seed = reinforceflow.get_random_seed()
        if seed and hasattr(self.env, 'seed'):
            self.env.seed(seed)

    def _step(self, action):
        gym_action = self._action_to_gym(action)
        obs, reward, done, info = self.env.step(gym_action)
        return self._obs_to_rf(obs), reward, done, info

    def _reset(self):
        obs = self._obs_to_rf(self.env.reset())
        return obs


@renewable
class AtariWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 start_action=None,
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
        self.env = GymWrapper(self.env)
        if clip_rewards:
            self.env = RewardClipWrap(self.env)
        if start_action:
            self.env = FireResetWrap(self.env, start_action=start_action)
        if noop_action:
            self.env = RandomNoOpWrap(self.env, noop_action=noop_action)
        self.env = ImageWrap(self.env, to_gray=to_gray, new_width=new_width, new_height=new_height)
        self.env = ActionRepeatWrap(self.env, action_repeat=action_repeat)
        self.env = ObservationStackWrap(self.env, obs_stack=obs_stack)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range


@renewable
class ImageWrap(gym.Wrapper):
    def __init__(self, env, to_gray=False, new_width=None, new_height=None):
        super(ImageWrap, self).__init__(env=env)
        self._height = new_height
        self._width = new_width
        self._to_gray = to_gray
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        new_shape = list(self.observation_space.shape)
        assert isinstance(self.observation_space, Continious) and 2 <= len(new_shape) <= 3,\
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


@renewable
class ActionRepeatWrap(gym.Wrapper):
    """
    Args:
        action_repeat (int): The number of steps on which the action will be repeated.
    """
    def __init__(self, env, action_repeat):
        super(ActionRepeatWrap, self).__init__(env=env)
        assert action_repeat > 0, "Action repeat number must be higher than 0."
        self._action_repeat = action_repeat

    def _step(self, action):
        obs, reward_total, done, info = self.env.step(action)
        for _ in range(self._action_repeat - 1):
            obs, reward, done, info = self.env.step(action)
            reward_total += reward
            if done:
                break
        return obs, reward_total, done, info


@renewable
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
        self._obs_stack_len = obs_stack or 1
        self.observation_space = self.env.observation_space
        new_shape = list(self.observation_space.shape)
        new_shape[-1] = self.observation_space.shape[-1] * obs_stack
        self.observation_space.reshape(tuple(new_shape))
        self._obs_stack = None

    def _reset(self):
        obs = self.env.reset()
        self._obs_stack = stack_observations(obs, self._obs_stack_len, self._obs_stack)
        return self._obs_stack

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._obs_stack = stack_observations(obs, self._obs_stack_len, self._obs_stack)
        return self._obs_stack, reward, done, info

    @property
    def obs_stack(self):
        """Observation stack length."""
        return self._obs_stack_len


@renewable
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

    def _reset(self, **kwargs):
        obs = self.env.reset()
        skip = np.random.randint(self._noop_min, self._noop_max)
        for _ in range(skip):
            obs, _, done, _ = self.env.step(self._noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


@renewable
class FireResetWrap(gym.Wrapper):
    def __init__(self, env, start_action):
        super(FireResetWrap, self).__init__(env=env)
        assert self.action_space.contains(start_action),\
            "Invalid action %s for %s environment." % (start_action, self.env)
        self._start_action = start_action

    def _reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(self._start_action)
        if done:
            obs = self.env.reset(**kwargs)
        return obs


@renewable
class ALELifeResetEnv(FireResetWrap):
    def __init__(self, env, start_action):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(ALELifeResetEnv, self).__init__(env=env, start_action=start_action)
        self.env = FireResetWrap(self.env, start_action=start_action)
        self._lives = 0
        self._needs_reset = True
        self._last_obs = None

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._needs_reset = done
        lives = self.env.unwrapped.ale.lives()
        if self._lives > lives > 0:
            done = True
        self._lives = lives
        return obs, reward, done, info

    def _reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self._needs_reset:
            self._last_obs = self.env.reset()
        elif self._start_action:
            self._last_obs, _, _, _ = self.env.step(self._start_action)
        self._lives = self.env.unwrapped.ale.lives()
        return self._last_obs


@renewable
class RewardClipWrap(gym.RewardWrapper):
    def _reward(self, reward):
        """Clips reward into {-1, 0, 1} range, as suggested in Mnih et al., 2013."""
        return np.sign(reward)


def _to_rf_space(space):
    """Converts Gym space instance into ReinforceFlow."""
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
            converted_spaces.append(_to_rf_space(sub_space))
        return Tuple(*converted_spaces)
    raise ValueError("Unsupported space %s." % space)


def _make_gym2rf_converter(space):
    """Makes converter function that maps space samples Gym -> ReinforceFlow."""
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
            sub_converters.append(_make_gym2rf_converter(sub_space))

        def converter(sample):
            converted_tuple = []
            for sub_sample, sub_converter in zip(sample, sub_converters):
                converted_tuple.append(sub_converter(sub_sample))
            return tuple(converted_tuple)
        return converter
    raise ValueError("Unsupported space %s." % space)


def _make_rf2gym_converter(space):
    """Makes space converter function that maps space samples ReinforceFlow -> Gym."""
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
            sub_converters.append(_make_rf2gym_converter(sub_space))

        def converter(sample):
            converted_tuple = []
            for sub_sample, sub_converter in zip(sample, sub_converters):
                converted_tuple.append(sub_converter(sub_sample))
            return tuple(converted_tuple)
        return converter
    raise ValueError("Unsupported space %s." % space)
