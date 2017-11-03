from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import abc
import six
from six.moves import range
from reinforceflow.utils import stack_observations
from reinforceflow.core import Tuple


@six.add_metaclass(abc.ABCMeta)
class Env(object):
    def __init__(self,
                 env,
                 obs_space,
                 action_space,
                 action_repeat=1,
                 obs_stack=1):
        """Base environment interface.

        In order to wrap an existing environment or create a new one,
        the following methods must be implemented:
            _step
            _reset
            (Optional) render

        Args:
            env: Raw environment instance.
            action_repeat (int): The number of steps on which the action will be repeated.
                           To disable, pass 1, 0 or None.
            obs_space: (core.spaces.Space) Observation space specification.
            action_space: (core.spaces.Space) Action space specification.
            obs_stack (int): The length of stacked observations.
                       Provided obs_space shape will be automatically modified.
                       Doesn't works for Tuple spaces. To disable, pass 1, 0 or None.
        """
        self.env = env
        self.obs_space = obs_space
        self.action_space = action_space
        if action_repeat < 0:
            raise ValueError("Action repeat number must be higher or equal to 0.")
        if obs_stack < 0:
            raise ValueError("Observation stack length must be higher or equal to 0.")
        self._action_repeat = action_repeat or 1
        self._obs_stack_len = obs_stack or 1
        if isinstance(self.obs_space, Tuple) and obs_stack > 1:
            raise ValueError("Observation stack does not works for Tuple spaces.")
        new_shape = list(self.obs_space.shape)
        if obs_stack > 1:
            new_shape[-1] = self.obs_space.shape[-1] * obs_stack
            self.obs_space.reshape(tuple(new_shape))
        self._obs_stack = None

    def _step(self, action):
        """See `step`."""
        raise NotImplementedError

    def _reset(self):
        """See `reset`."""
        raise NotImplementedError

    def render(self):
        """Renders current observation."""
        raise NotImplementedError

    def reset(self):
        """Resets current episode.

        Returns (numpy.ndarray):
            First observation from the new episode.
        """
        # Reset observations stack
        obs = self._reset()
        self._obs_stack = stack_observations(obs, self._obs_stack_len, self._obs_stack)
        return self._obs_stack

    def step(self, action):
        """Executes step with given action.

        Args:
            action (numpy.ndarray): Action for current step.
        Returns:
            Transition tuple of (next_observation, reward, is_terminal, info).
        """
        reward_total = 0
        done = False
        # Action repeat
        for _ in range(self._action_repeat):
            obs, reward, done, info = self._step(action)
            reward_total += reward
            if done:
                break
        # Observation stacking
        self._obs_stack = stack_observations(obs, self._obs_stack_len, self._obs_stack)
        return self._obs_stack, reward_total, done, info

    def copy(self):
        return copy.deepcopy(self)

    @property
    def obs_stack(self):
        """Observation stack length."""
        return self._obs_stack_len
