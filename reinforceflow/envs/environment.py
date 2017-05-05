from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from reinforceflow.envs.env_wrapper import EnvWrapper


class MountainCarSimplified(EnvWrapper):
    def __init__(self, env):
        super(MountainCarSimplified, self).__init__(env)

    def step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        if obs_next[1] > 0:
            reward += obs_next[1] * 10
        return self._obs2vec(obs_next), reward, done, info


def show(frame, iframe=0, thread=0, path='images'):
    if not os.path.exists(path):
        os.makedirs(path)
    #with lock:
    batch_size = frame.shape[0]
    stack_size = frame.shape[3]
    for batch in range(batch_size):
        for ch in range(stack_size):
            plt.subplot(batch_size, stack_size, batch*stack_size + (ch+1))
            plt.imshow(frame[batch][:,:,ch], cmap='gray', interpolation='nearest')
            plt.axis('off')
    plt.savefig('%s/%d_%d.png' % (path, iframe, thread), bbox_inches='tight')


class SimpleEnv:
    def __init__(self, w=40, h=40, actrep=1, memlen=1, total_frames=100, name='Test', render=False):
        """Maximum episode reward from -`total_frames` to + `total_frames`"""
        self.memlen = memlen
        self.W = w
        self.H = h
        self.actrep = actrep
        self.stacked_s = None
        self._action_space = [0, 1]
        self._action_size = len(self._action_space)
        self.render = render
        self._last_label = 0
        self._frame_counter = 0
        self._total_frames = total_frames
        self.observation_space = spaces.Box(low=0, high=1, shape=[self.W, self.H, 1])
        self.action_space = spaces.Discrete(2)

    def preprocess(self, s):
        s = s.reshape(1, s.shape[0], s.shape[1], 1)
        if self.stacked_s is not None:
            self.stacked_s = np.append(s, self.stacked_s[:, :, :, :self.memlen - 1], axis=3)
        else:
            self.stacked_s = np.repeat(s, self.memlen, axis=3)
        return self.stacked_s

    def reset(self):
        self.stacked_s = None
        return self.preprocess(self._reset())

    def reset_random(self):
        self.stacked_s = None
        return self.preprocess(self._reset())

    def step(self, action, test=False):
        """Executes action and repeat it on the next X frames
        :param action one-hot encoded action (e.g. [0, 1, 0])
        :rtype 4 elements tuple:
               new state,
               accumulated reward over skipped frames
               is terminal,
               info"""
        action = self._action_space[action]
        accum_reward = 0
        for _ in range(self.actrep):
            s, r, term, info = self._step(action)
            accum_reward += r
            if term:
                break
        if self.render:
            self._render()
        return self.preprocess(s), accum_reward, term, info

    def _next_state(self):
        self._last_label = random.randrange(self._action_size)
        if self._last_label == 0:
            state = np.zeros((self.W, self.H)).astype('float32')
        else:
            state = np.ones((self.W, self.H)).astype('float32')
        return state

    def _reset(self):
        self._frame_counter = 0
        return self._next_state()

    def _step(self, action):
        if self._frame_counter >= self._total_frames:
            raise Exception('A reset is required. Frame: %s; Total frames: %s' % (self._frame_counter, self._total_frames))
        reward = -1
        if action == self._last_label:
            reward = 1
        state = self._next_state()
        self._frame_counter += 1
        term = self._frame_counter == self._total_frames
        return state, reward, term, {}

    def _render(self):
        raise NotImplementedError


class DelayedEnv(SimpleEnv):
    def __init__(self, w=40, h=40, actrep=1, memlen=4, total_frames=10, name='Test', render=False):
        """Maximum episode reward from -`total_frames` to + `total_frames`"""
        SimpleEnv.__init__(self, w, h, actrep, memlen, total_frames, name, render)
        self.episode_reward = 0

    def step(self, action, test=False):
            """Executes action and repeat it on the next X frames
            :param action one-hot encoded action (e.g. [0, 1, 0])
            :rtype 4 elements tuple:
                   new state,
                   accumulated reward over skipped frames
                   is terminal,
                   info"""
            action = self._action_space[action]
            accum_reward = 0
            for _ in range(self.actrep):
                s, r, term, info = self._step(action)
                self.episode_reward += r
                if term:
                    accum_reward = self.episode_reward
                    self.episode_reward = 0
                    break
            if self.render:
                self._render()
            return self.preprocess(s), accum_reward, term, info
