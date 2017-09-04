from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
from six.moves import range
from reinforceflow.utils import stack_observations


class EnvWrapper(object):
    def __init__(self,
                 env,
                 continious_action,
                 continious_observation,
                 action_repeat=1,
                 obs_stack=1):
        """Base environment interface.

        In order to wrap a custom environment or create a new one,
        the following methods must be implemented:
            _step
            _reset
            action_sample

        Args:
            env: Raw environment instance.
            continious_action: (bool) Whether action space is continious.
            continious_observation: (bool) Whether observation space consists
                                    from continious values. If true, table-based agents
                                    won't be able to use this environment.
            action_repeat: (int) The number of steps on which the action will be repeated.
                           To disable, pass 1, 0 or None.
            obs_stack: (int) The length of stacked observations.
                       Used for providing a short-term memory. To disable, pass 1, 0 or None.
        """
        self.env = env
        self.is_cont_action = continious_action
        self.is_cont_obs = continious_observation
        if action_repeat < 0:
            raise ValueError("Action repeat number must be higher or equal to 0.")
        if obs_stack < 0:
            raise ValueError("Observation stack length must be higher or equal to 0.")
        self._action_repeat = action_repeat or 1
        self._obs_stack_len = obs_stack or 1
        self._obs_stack = None
        self.obs_shape = list(np.shape(self.reset()))
        self.action_shape = list(np.shape(self.action_sample()))
        self.is_multiaction = len(self.action_shape) > 1

    def _step(self, action):
        """See `step`."""
        raise NotImplementedError

    def _reset(self):
        """See `reset`."""
        raise NotImplementedError

    def action_sample(self):
        """Samples random action from environment's action space."""
        raise NotImplementedError

    def reset(self):
        """Resets current episode.

        Returns: (nd.array)
            First observation from the new episode.
        """
        # Reset observations stack
        self._obs_stack = None
        obs = self._reset()
        if self._obs_stack_len > 1:
            obs = stack_observations(obs, self._obs_stack_len, self._obs_stack)
        return obs

    def step(self, action):
        """Executes step with given action.

        Args:
            action: (nd.array) Action for current step.
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
        if self._obs_stack_len and self._obs_stack_len > 1:
            obs = stack_observations(obs, self._obs_stack_len, self._obs_stack)
        return obs, reward_total, done, info

    def copy(self):
        return copy.deepcopy(self)
