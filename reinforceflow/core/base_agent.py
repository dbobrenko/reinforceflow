from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import defaultdict
import abc
import six
import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

import reinforceflow.utils
from reinforceflow.core import GreedyPolicy
from reinforceflow import logger


@six.add_metaclass(abc.ABCMeta)
class BaseAgent(object):
    @abc.abstractmethod
    def __init__(self, env):
        super(BaseAgent, self).__init__()
        self.env = env

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def predict_action(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, episodes, max_ep_steps=int(1e5), render=False, copy_env=False):
        """Tests agent's performance with specified policy on a given number of episodes.

        Args:
            episodes: (int) Number of episodes.
            max_ep_steps: (int) Maximum allowed steps per episode.
            render: (bool) Enables game screen rendering.
            copy_env: (bool) Performs tests on the copy of environment instance.

        Returns: (utils.IncrementalAverage) Average reward per episode.
        """
        env = self.env.copy() if copy_env else self.env
        ep_rewards = reinforceflow.utils.IncrementalAverage()
        for _ in range(episodes):
            reward_accum = 0
            obs = env.reset()
            for _ in range(max_ep_steps):
                if render:
                    env.render()
                action = self.predict_action(obs)
                obs, r, terminal, info = env.step(action)
                reward_accum += r
                if terminal:
                    break
            ep_rewards.add(reward_accum)
        return ep_rewards


@six.add_metaclass(abc.ABCMeta)
class BaseDiscreteAgent(BaseAgent):
    """Base class for Agent with discrete action space."""
    @abc.abstractmethod
    def __init__(self, env):
        super(BaseDiscreteAgent, self).__init__(env)
        if self.env.is_cont_action:
            raise ValueError('%s does not support environments with continuous '
                             'action space.' % self.__class__.__name__)
        if self.env.is_multiaction:
            raise ValueError('%s does not support environments with multiple '
                             'action spaces.' % self.__class__.__name__)


@six.add_metaclass(abc.ABCMeta)
class BaseTableAgent(BaseDiscreteAgent):
    """Base class for Table-based Agent with discrete observation and action space."""
    @abc.abstractmethod
    def __init__(self, env):
        super(BaseTableAgent, self).__init__(env)
        if self.env.is_cont_obs:
            raise ValueError('%s does not support environments with continuous '
                             'observation space.' % self.__class__.__name__)
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_shape))


@six.add_metaclass(abc.ABCMeta)
class BaseDeepAgent(BaseAgent):
    @abc.abstractmethod
    def __init__(self, env, net_factory, name=''):
        """Abstract base class for Deep Q-Network agent.

        Args:
            env: Environment wrapper instance.
            net_factory: Network factory, defined in nets file.

        Attributes:
            env: Environment instance.
            net_factory: (function) Used for building network model.
            name: (str) Agent's name prefix.
        """
        super(BaseDeepAgent, self).__init__(env=env)
        self._net_factory = net_factory
        self._scope = '' if not name else name + '/'
        # Inference Graph
        with tf.variable_scope(self._scope + 'network') as scope:
            self._action_ph = tf.placeholder('int32', [None] + self.env.action_shape, name='action')
            self._reward_ph = tf.placeholder('float32', [None], name='reward')
            self.net = self._net_factory.make(input_shape=[None] + self.env.obs_shape,
                                              output_size=self.env.action_shape[0])
            self._weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope.name)
        with tf.variable_scope(self._scope + 'optimizer'):
            self._no_op = tf.no_op()
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self._ep_counter = tf.Variable(0, trainable=False, name='ep_counter')
            self._ep_counter_inc = self._ep_counter.assign_add(1, use_locking=True)
            self._obs_counter = tf.Variable(0, trainable=False, name='obs_counter')
            self._obs_counter_inc = self._obs_counter.assign_add(1, use_locking=True)
        self.sess = None
        self._saver = None

    def build_train_graph(self, optimizer, learning_rate, optimizer_args=None,
                          decay=None, decay_args=None, gradient_clip=40.0, saver_keep=3):
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
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError

    def train_on_batch(self, *args, **kwargs):
        raise NotImplementedError

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
        logger.info('Checkpoint has been restored from: %s', checkpoint)

    @property
    def ep_counter(self):
        return self.sess.run(self._ep_counter)

    def increment_ep_counter(self):
        return self.sess.run(self._ep_counter_inc)

    @property
    def obs_counter(self):
        return self.sess.run(self._obs_counter)

    def increment_obs_counter(self):
        return self.sess.run(self._obs_counter_inc)

    @property
    def step_counter(self):
        return self.sess.run(self.global_step)

    def predict_on_batch(self, obs_batch):
        """Computes action-values for given batch of observations."""
        return self.sess.run(self.net.output, {self.net.input_ph: obs_batch})

    def close(self):
        if self.sess:
            self.sess.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@six.add_metaclass(abc.ABCMeta)
class BaseDQNAgent(BaseDeepAgent, BaseDiscreteAgent):
    def __init__(self, env, net_factory, name=''):
        super(BaseDQNAgent, self).__init__(env, net_factory, name)
        self._target_net = None
        self._target_update = None
        self._greedy_policy = GreedyPolicy()

    def load_weights(self, checkpoint):
        super(BaseDQNAgent, self).load_weights(checkpoint)
        self.target_update()

    def predict_action(self, obs):
        """Computes action with greedy policy for given observation."""
        action_values = self.predict_on_batch([obs])
        return self._greedy_policy.select_action(self.env, action_values)

    def target_predict(self, obs):
        """Computes target network action-values with for given batch of observations."""
        return self.sess.run(self._target_net.output, {self._target_net.input_ph: obs})

    def target_update(self):
        """Updates target network."""
        self.sess.run(self._target_update)
