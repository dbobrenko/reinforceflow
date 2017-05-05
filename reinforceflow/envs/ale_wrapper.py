from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from reinforceflow.envs.env_wrapper import EnvWrapper
from reinforceflow.core import AtariPreprocessor


class ALEWrapper(EnvWrapper):
    def __init__(self, env, preprocessor=AtariPreprocessor(stack_len=4), action_repeat=0, random_start=0):
        super(ALEWrapper, self).__init__(env, preprocessor, action_repeat, random_start)
        self.has_lives = hasattr(self.env, 'ale') and hasattr(self.env.ale, 'lives')

    def step(self, action):
        if self.has_lives:
            start_lives = self.env.ale.lives()
        stack_reset = self._needs_stack_reset
        self._needs_stack_reset = False
        reward_accum = 0
        for _ in range(self._action_repeat):
            obs, reward, done, info = self.env.step(action)
            reward_accum += reward
            if self.has_lives and self.env.ale.lives() < start_lives:
                self._needs_stack_reset = True
                break
            if done:
                self._needs_stack_reset = True
                break
        # Takes maximum value for each pixel value over the current and previous frame
        # Used to get round Atari sprites flickering (Mnih et al. (2015))
        return self.preprocessor(self._obs2vec(obs), reset_stack=stack_reset), reward_accum, done, info
