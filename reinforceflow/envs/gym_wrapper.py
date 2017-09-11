from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
try:
    import gym
    from gym import spaces
except ImportError:
    gym = None

import numpy as np
import reinforceflow
from reinforceflow.envs.env_wrapper import EnvWrapper
from reinforceflow.utils import stack_observations, image_preprocess, one_hot


class GymWrapper(EnvWrapper):
    """Light wrapper around OpenAI Gym and Universe environments.
    See `EnvWrapper`.
    """
    def __init__(self, env, action_repeat=1, obs_stack=1):
        if gym is None:
            raise ImportError("Cannot import OpenAI Gym. In order to use Gym environments "
                              "you must install it first. Follow the instructions on "
                              "https://github.com/openai/gym#installation")
        if isinstance(env, six.string_types):
            env = gym.make(env)
        if isinstance(env.action_space, spaces.Tuple):
            raise ValueError("Gym environments with tuple spaces aren't supported yet.")
        continious_action = isinstance(env.action_space, spaces.Box)
        continious_observation = isinstance(env.observation_space, spaces.Box)
        super(GymWrapper, self).__init__(env,
                                         continious_action=continious_action,
                                         continious_observation=continious_observation,
                                         action_repeat=action_repeat,
                                         obs_stack=obs_stack)
        seed = reinforceflow.get_random_seed()
        if seed and hasattr(self.env, 'seed'):
            self.env.seed(seed)

    def _step(self, action):
        gym_action = self._rf_to_gym(action, self.env.action_space)
        obs, reward, done, info = self.env.step(gym_action)
        rf_obs = self._gym_to_rf(obs, self.env.observation_space)
        return rf_obs, reward, done, info

    def _reset(self):
        return self._gym_to_rf(self.env.reset(), self.env.observation_space)

    def action_sample(self):
        return self._gym_to_rf(self.env.action_space.sample(), self.env.action_space)

    @classmethod
    def _gym_to_rf(cls, sample, space_type):
        if isinstance(space_type, spaces.Box):
            return sample
        elif isinstance(space_type, spaces.Discrete):
            return one_hot(space_type.n, sample)
        else:
            raise ValueError("Unsupported Gym space: %s." % space_type)

    @classmethod
    def _rf_to_gym(cls, sample, space_type):
        if isinstance(space_type, spaces.Box):
            return sample
        elif isinstance(space_type, spaces.Discrete):
            return np.argmax(sample)
        else:
            raise ValueError("Unsupported Gym space: %s." % space_type)


class GymPixelWrapper(GymWrapper):
    def __init__(self,
                 env,
                 action_repeat,
                 obs_stack,
                 to_gray=False,
                 resize_width=None,
                 resize_height=None,
                 merge_last_frames=False):
        self.height = resize_height
        self.width = resize_width
        self.to_gray = to_gray
        self._use_merged_frame = merge_last_frames
        super(GymPixelWrapper, self).__init__(env,
                                              action_repeat=action_repeat,
                                              obs_stack=obs_stack)
        if len(self.obs_shape) not in [2, 3]:
            raise ValueError('%s expects observation space with pixel inputs.'
                             % self.__class__.__name__)
        self.has_lives = hasattr(self.env, 'ale') and hasattr(self.env.ale, 'lives')
        self._prev_obs = None

    def step(self, action):
        """See `EnvWrapper.step`."""
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
        return self._gym_to_rf(self._obs_preprocess(self.env.reset()),
                               self.env.observation_space)

    def _obs_preprocess(self, obs):
        """Applies such image preprocessing as resizing and converting to grayscale.
        Also, takes maximum value for each pixel value over the current and previous frame.
        Used to get around Atari sprites flickering (see Mnih et al. (2015)).

        Args:
            obs: (nd.array) 2-D or 3-D observation.
        Returns:
            (nd.array) Preprocessed 3-D observation.
        """
        obs = image_preprocess(obs, resize_height=self.height,
                               resize_width=self.width, to_gray=self.to_gray)
        if self._use_merged_frame and self._prev_obs is not None:
            prev_obs = self._prev_obs
            self._prev_obs = obs
            obs = np.maximum.reduce([obs, prev_obs]) if prev_obs else obs
        return obs
