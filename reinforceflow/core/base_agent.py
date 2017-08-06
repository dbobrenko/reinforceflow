from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import defaultdict
from abc import ABCMeta, abstractmethod

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

from reinforceflow import error
from reinforceflow.envs.env_wrappers import EnvWrapper
from reinforceflow.core import GreedyPolicy
from reinforceflow import misc
from reinforceflow import logger


class BaseAgent(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, env):
        super(BaseAgent, self).__init__()
        if not isinstance(env, EnvWrapper):
            logger.warn("Wrapping environment %s into EnvWrapper." % env)
            env = EnvWrapper(env)
        self.env = env

    def train(self, *args, **kwargs):
        pass


class BaseDiscreteAgent(BaseAgent):
    """Base class for Agent with discrete action space."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, env):
        super(BaseDiscreteAgent, self).__init__(env)
        if self.env.is_cont_action:
            raise error.UnsupportedSpace('%s does not support environments with continuous '
                                         'action space.' % self.__class__.__name__)
        if self.env.has_multiple_action:
            raise error.UnsupportedSpace('%s does not support environments with multiple '
                                         'action spaces.' % self.__class__.__name__)


class TableAgent(BaseDiscreteAgent):
    """Base class for Table-based Agent with discrete observation and action space."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, env):
        super(TableAgent, self).__init__(env)
        if self.env.is_cont_obs:
            raise error.UnsupportedSpace('%s does not support environments with continuous '
                                         'observation space.' % self.__class__.__name__)
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_shape))


class BaseDQNAgent(BaseDiscreteAgent):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, env, net_fn, name=''):
        """Abstract base class for Deep Q-Network agent.

        Args:
            env (envs.RawGymWrapper): Environment wrapper.
            net_fn: Function, takes `input_shape` and `output_size` arguments,
                    returns tuple(input Tensor, output Tensor, all end point Tensors).

        Attributes:
            env: Current environment.
            net_fn: Function, used for building network model.
            name: Agent's name prefix.
        """
        super(BaseDQNAgent, self).__init__(env=env)
        self.net_fn = net_fn
        self._scope = '' if not name else name + '/'
        self.global_step = None
        self._obs_counter = None
        self._no_op = tf.no_op()
        self.opt = None
        self.sess = None
        self._action = None
        self._reward = None
        self._obs = None
        self._q = None
        self._weights = None
        self._target_obs = None
        self._target_q = None
        self._target_weights = None
        self._target_update = None
        self._lr = None
        self._action_one_hot = None
        self._loss = None
        self._grads = None
        self._grads_vars = None
        self._train_op = None
        self._saver = None
        self._summary_op = None
        self._init_op = None
        self._save_vars = set()

    def _build_inference_graph(self, env):
        if self._q is not None:
            logger.warn("The inference graph has already been built. Skipping..")
            return
        with tf.variable_scope(self._scope + 'network') as scope:
            self._action = tf.placeholder('int32', [None], name='action')
            self._reward = tf.placeholder('float32', [None], name='reward')
            self._obs, self._q, _ = self.net_fn(input_shape=[None] + env.observation_shape,
                                                output_size=env.action_shape)
            self._weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope.name)

    def _build_train_graph(self, optimizer, learning_rate, optimizer_args=None,
                           decay=None, decay_args=None, gradient_clip=40.0, saver_keep=10):
        """Builds training graph.

        Args:
            optimizer: An optimizer name string or class.
            learning_rate (float or Tensor): Optimizer's learning rate.
            optimizer_args (dict): Keyword arguments used for optimizer creation.
            decay (function): Learning rate decay.
                              Expects tensorflow decay function or function name string.
                              Available name strings: 'polynomial', 'exponential'.
                              To disable, pass None.
            decay_args (dict): Keyword arguments, passed to the decay function.
            gradient_clip (float): Norm gradient clipping.
                                   To disable, pass False or None.
            saver_keep (int): Maximum number of checkpoints can be stored in `log_dir`.
                              When exceeds, overwrites the most earliest checkpoints.
        """
        if self._train_op is not None:
            logger.warn("The training graph has already been built. Skipping.")
            return
        with tf.variable_scope(self._scope + 'target_network') as scope:
            self._target_obs, self._target_q, _ = \
                self.net_fn(input_shape=[None] + self.env.observation_shape,
                            output_size=self.env.action_shape)
            self._target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope.name)
            self._target_update = [self._target_weights[i].assign(self._weights[i])
                                   for i in range(len(self._target_weights))]

        with tf.variable_scope(self._scope + 'optimizer'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self._obs_counter = tf.Variable(0, trainable=False, name='obs_counter')
            self._obs_counter_inc = self._obs_counter.assign_add(1, use_locking=True)
            self.opt, self._lr = misc.create_optimizer(optimizer, learning_rate,
                                                       optimizer_args=optimizer_args,
                                                       decay=decay, decay_args=decay_args,
                                                       global_step=self.global_step)
            self._action_one_hot = tf.one_hot(self._action, self.env.action_shape, 1.0, 0.0,
                                              name='action_one_hot')
            # Predict expected future reward for performed action
            q_value = tf.reduce_sum(tf.multiply(self._q, self._action_one_hot), axis=1)
            self._td_error = self._reward - q_value
            self._loss = tf.reduce_mean(tf.square(self._td_error), name='loss')
            self._grads = tf.gradients(self._loss, self._weights)
            if gradient_clip:
                self._grads, _ = tf.clip_by_global_norm(self._grads, gradient_clip)
            self._grads_vars = list(zip(self._grads, self._weights))
            self._train_op = self.opt.apply_gradients(self._grads_vars,
                                                      global_step=self.global_step)
        self._save_vars |= set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 self._scope + 'network'))
        self._save_vars |= set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 self._scope + 'optimizer'))

        self._save_vars.add(self.global_step)
        self._save_vars.add(self._obs_counter)
        self._saver = tf.train.Saver(var_list=list(self._save_vars), max_to_keep=saver_keep)
        self._summary_op = tf.no_op()

    def test(self, episodes, policy=GreedyPolicy(), max_ep_steps=int(1e5), render=False):
        """Tests agent's performance with specified policy on a given number of episodes.

        Args:
            episodes (int): Number of episodes.
            policy (core.BasePolicy): Agent's policy.
            max_ep_steps (int): Maximum allowed steps per episode.
            render (bool): Enables game screen rendering.

        Returns (tuple): Average reward per episode, average max. Q value per episode.
        """
        ep_rewards = misc.IncrementalAverage()
        ep_q = misc.IncrementalAverage()
        for _ in range(episodes):
            reward_accum = 0
            obs = self.env.reset()
            for _ in range(max_ep_steps):
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

    def train(self, **kwargs):
        raise NotImplementedError

    def train_on_batch(self, obs, actions, rewards, summarize=False):
        """Trains agent on given transitions batch.

        Args:
            obs (nd.array): Input observations with shape=[batch, height, width, channels].
            actions (list): Actions.
            rewards (list): Rewards for each action.
            summarize (bool): Enables TensorBoard summary writing.
        """
        if self._train_op is not None:
            raise ValueError('Consider calling `_build_train_graph` '
                             'before starting training process.')

        return self._train_on_batch(obs, actions, rewards, summarize)

    def _train_on_batch(self, obs, actions, rewards, summarize=False):
        _, td_error, summary = self.sess.run([self._train_op, self._td_error,
                                              self._summary_op if summarize else self._no_op],
                                             feed_dict={
                                                 self._obs: obs,
                                                 self._action: actions,
                                                 self._reward: rewards
                                             })
        return td_error, summary

    def save_weights(self, path, model_name='model.ckpt'):
        if not os.path.exists(path):
            os.makedirs(path)
        self._saver.save(self.sess, os.path.join(path, model_name), global_step=self.global_step)
        logger.info('Checkpoint has been saved to: %s' % os.path.join(path, model_name))

    def load_weights(self, checkpoint):
        if not os.path.exists(checkpoint):
            raise ValueError('Checkpoint path/dir %s does not exists.' % checkpoint)
        if tf.gfile.IsDirectory(checkpoint):
            checkpoint = tf.train.latest_checkpoint(checkpoint)
        self._saver.restore(self.sess, save_path=checkpoint)
        self.target_update()
        logger.info('Checkpoint has been restored from: %s', checkpoint)

    def increment_obs_counter(self):
        return self.sess.run(self._obs_counter_inc)

    @property
    def obs_counter(self):
        return self.sess.run(self._obs_counter_inc)

    @property
    def step_counter(self):
        return self.sess.run(self.global_step)

    def predict(self, obs):
        return self.sess.run(self._q, {self._obs: obs})

    def target_predict(self, obs):
        return self.sess.run(self._target_q, {self._target_obs: obs})

    def target_update(self):
        self.sess.run(self._target_update)

    def close(self):
        if self.sess:
            self.sess.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
