from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import os
from collections import defaultdict

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

from reinforceflow import error
from reinforceflow.envs.env_wrapper import EnvWrapper
from reinforceflow.core import ExperienceReplay
from reinforceflow.nets import dqn
from reinforceflow.core import EGreedyPolicy, GreedyPolicy
from reinforceflow import misc
from reinforceflow import logger


class BaseAgent(object):
    def __init__(self, env):
        if not isinstance(env, EnvWrapper):
            env = EnvWrapper(env)
        self.env = env

    def train(self, *args, **kwargs):
        pass


class BaseDiscreteAgent(BaseAgent):
    """Base class for Agent with discrete action space"""
    def __init__(self, env):
        super(BaseDiscreteAgent, self).__init__(env)
        if self.env.is_cont_action:
            raise error.UnsupportedSpace('%s does not support environments with continuous action space.'
                                         % self.__class__.__name__)
        if self.env.has_multiple_action:
            raise error.UnsupportedSpace('%s does not support environments with multiple action spaces.'
                                         % self.__class__.__name__)
        self.env = env


class TableAgent(BaseDiscreteAgent):
    """Base class for Table-based Agent with discrete observation and action space"""
    def __init__(self, env):
        super(TableAgent, self).__init__(env)
        if self.env.is_cont_obs:
            raise error.UnsupportedSpace('%s does not support environments with continuous observation space.'
                                         % self.__class__.__name__)
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_shape))


class BaseDQNAgent(BaseDiscreteAgent):
    def __init__(self,
                 env,
                 net_fn,
                 optimizer,
                 learning_rate=0.00025,
                 optimizer_args=None,
                 gradient_clip=40.0,
                 decay=None,
                 decay_args=None):
        """Base class for Deep Q-Network agent.

        Args:
            env (reinforceflow.EnvWrapper): Environment wrapper.
            net_fn: Function, that takes `input_shape` and `output_size` arguments,
                    and returns tuple(input Tensor, output Tensor, all end point Tensors).
            optimizer: An optimizer string name or class.
            learning_rate (float or Tensor): Optimizer's learning rate.
            optimizer_args (dict): keyword arguments, used for chosen tensorflow optimizer creation.
            gradient_clip (float): Norm gradient clipping, to disable, pass False or None.
            decay: Learning rate decay. Should be provided decay function, or decay function name.
                   Available decays: 'polynomial', 'exponential'. To disable decay, pass None.
            decay_args (dict): keyword arguments, passed to the chosen tensorflow learning rate decay function.

        Attributes:
            env: Current environment
            optimizer: TensorFlow optimizer
        """
        super(BaseDQNAgent, self).__init__(env=env)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        with tf.variable_scope('network'):
            self._action = tf.placeholder('int32', [None], name='action')
            self._reward = tf.placeholder('float32', [None], name='reward')
            self._obs, self._q, _ = net_fn(input_shape=[None] + self.env.observation_shape,
                                           output_size=self.env.action_shape)

        with tf.variable_scope('target_network'):
            self._target_obs, self._target_q, _ = net_fn(input_shape=[None] + self.env.observation_shape,
                                                         output_size=self.env.action_shape)

        with tf.variable_scope('target_update'):
            target_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_network')
            w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'network')
            self._target_update = [target_w[i].assign(w[i]) for i in range(len(target_w))]

        with tf.variable_scope('optimizer'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.opt, self.lr = misc.create_optimizer(optimizer, learning_rate, optimizer_args=optimizer_args,
                                                      decay=decay, decay_args=decay_args, global_step=self.global_step)
            self._action_one_hot = tf.one_hot(self._action, self.env.action_shape, 1.0, 0.0, name='action_one_hot')
            # Predict expected future reward for performed action
            q_value = tf.reduce_sum(tf.multiply(self._q, self._action_one_hot), axis=1)
            self._loss = tf.reduce_mean(tf.square(self._reward - q_value), name='loss')
            grads = tf.gradients(self._loss, w)
            if gradient_clip:
                grads, _ = tf.clip_by_global_norm(grads, gradient_clip)
            self._grads_vars = list(zip(grads, w))
            self._train_op = self.opt.apply_gradients(self._grads_vars, global_step=self.global_step)
        self._saver = tf.train.Saver(max_to_keep=10)
        self._summary_op = tf.no_op()

    @property
    def current_step(self):
        return self.sess.run(self.global_step)

    def predict(self, obs):
        return self.sess.run(self._q, {self._obs: obs})

    def predict_target(self, obs):
        return self.sess.run(self._target_q, {self._target_obs: obs})

    def update_target(self):
        self.sess.run(self._target_update)

    def train_on_batch(self, obs, actions, rewards, summarize=False):
        """Trains agent on given transitions batch.

        Args:
            obs (nd.array): input observations with shape=[batch, height, width, channels]
            actions: list of actions
            rewards: list with rewards for each action
            summarize: if enabled, writes summaries into TensorBoard
        """
        _, summary = self.sess.run([self._train_op, self._summary_op if summarize else tf.no_op()], feed_dict={
                                    self._obs: obs,
                                    self._action: actions,
                                    self._reward: rewards
                                    })
        return summary

    def test(self, episodes, policy=GreedyPolicy(), max_ep_steps=1e5, render=False):
        """Tests agent's performance with specified policy on given number of games"""
        ep_rewards = misc.IncrementalAverage()
        ep_q = misc.IncrementalAverage()
        for _ in range(episodes):
            reward_accum = 0
            obs = self.env.reset()
            for _ in range(int(max_ep_steps)):
                if render:
                    self.env.render()
                reward_per_action = self.predict(obs)
                action = policy.select_action(self.env, reward_per_action)
                obs, r, terminal, info = self.env.step(action)
                ep_q.add(np.max(reward_per_action))
                reward_accum += r
                if terminal:
                    break
            ep_rewards.add(reward_accum)
        return ep_rewards.compute_average(), ep_q.compute_average()

    def _train(self, **kwargs):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError
