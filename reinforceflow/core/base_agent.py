from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import os
import time
from collections import defaultdict

import six
import tensorflow as tf
from six.moves import range  # pylint: disable=redefined-builtin

import reinforceflow.utils
from reinforceflow import logger
from reinforceflow.core import GreedyPolicy
from reinforceflow.core import Tuple, Continious


@six.add_metaclass(abc.ABCMeta)
class BaseAgent(object):
    @abc.abstractmethod
    def __init__(self, env, net_factory, name=''):
        """Abstract base class for Deep Network-based agents.

        Args:
            env (envs.Env): Environment instance.
            net_factory (nets.AbstractFactory): Network factory, defined in nets file.

        Attributes:
            env: Environment instance.
            net_factory: Used for building network model.
            name: Agent's name prefix.
        """
        super(BaseAgent, self).__init__()
        self.env = env
        self._net_factory = net_factory
        self._scope = '' if not name else name + '/'
        # Inference Graph
        with tf.variable_scope(self._scope + 'network') as scope:
            self._action_ph = tf.placeholder('int32', [None] + list(self.env.action_space.shape),
                                             name='action')
            self._reward_ph = tf.placeholder('float32', [None], name='reward')
            self.net = self._net_factory.make(input_space=self.env.obs_space,
                                              output_space=self.env.action_space)
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

    def predict_action(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, episodes, max_ep_steps=int(1e5), render=False, copy_env=False, max_fps=None):
        """Tests agent's performance with specified policy on a given number of episodes.

        Args:
            episodes (int): Number of episodes.
            max_ep_steps (int): Maximum allowed steps per episode.
            render (bool): Enables game screen rendering.
            copy_env (bool): Performs tests on the copy of environment instance.
            max_fps (int): Maximum allowed fps. To disable fps limitation, pass None.

        Returns (utils.IncrementalAverage): Average reward per episode.
        """
        # In seconds
        delta_frame = 1. / max_fps if max_fps else 0
        env = self.env.copy() if copy_env else self.env
        ep_rewards = reinforceflow.utils.IncrementalAverage()
        for _ in range(episodes):
            reward_accum = 0
            obs = env.reset()
            for _ in range(max_ep_steps):
                start_time = time.time()
                action = self.predict_action(obs)
                obs, r, terminal, info = env.step(action)
                reward_accum += r
                if render:
                    env.render()
                    if delta_frame > 0:
                        delay = max(0, delta_frame - (time.time() - start_time))
                        time.sleep(delay)
                if terminal:
                    break
            ep_rewards.add(reward_accum)
        return ep_rewards


@six.add_metaclass(abc.ABCMeta)
class BaseDiscreteAgent(BaseAgent):
    """Base class for Agent with discrete action space."""
    @abc.abstractmethod
    def __init__(self, env, net_factory, name=''):
        super(BaseDiscreteAgent, self).__init__(env=env, net_factory=net_factory, name=name)
        if isinstance(self.env.action_space, Continious):
            raise ValueError('%s does not support environments with continuous '
                             'action space.' % self.__class__.__name__)
        if isinstance(self.env.action_space, Tuple):
            raise ValueError('%s does not support environments with multiple '
                             'action spaces.' % self.__class__.__name__)


@six.add_metaclass(abc.ABCMeta)
class BaseTableAgent(BaseDiscreteAgent):
    """Base class for Table-based Agent with discrete observation and action space."""
    @abc.abstractmethod
    def __init__(self, env, net_factory, name=''):
        super(BaseTableAgent, self).__init__(env=env, net_factory=net_factory, name=name)
        if isinstance(self.env.obs_shape, Continious):
            raise ValueError('%s does not support environments with continuous '
                             'observation space.' % self.__class__.__name__)
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_shape))


@six.add_metaclass(abc.ABCMeta)
class BaseDQNAgent(BaseDiscreteAgent):
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


@six.add_metaclass(abc.ABCMeta)
class BaseAsyncAgent(BaseAgent):
    def __init__(self, env, net_factory, use_gpu=False, name=''):
        super(BaseAsyncAgent, self).__init__(env=env, net_factory=net_factory, name=name)
        config = tf.ConfigProto(
            device_count={'GPU': use_gpu}
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self._greedy_policy = GreedyPolicy()
        self.weights = self._weights
        self.request_stop = False
        self._target_update = None
        self._reward_logger = None
        self.writer = None
        self.opt = None

    def build_train_graph(self, optimizer, learning_rate, optimizer_args=None,
                          decay=None, decay_args=None, gradient_clip=40.0, saver_keep=10):
        """Builds training graph.

        Args:
            optimizer: An optimizer name string or class.
            learning_rate (float or Tensor): Optimizer's learning rate.
            optimizer_args (dict): Keyword arguments used for optimizer creation.
            decay (str or function): Learning rate decay.
                Expects tensorflow decay function or function name string.
                Available name strings: 'polynomial', 'exponential'.
                To disable, pass None.
            decay_args (dict): Keyword arguments, passed to the decay function.
            gradient_clip (float): Norm gradient clipping.
                To disable, pass False or None.
            saver_keep (int): Maximum number of checkpoints can be stored in `log_dir`.
                When exceeds, overwrites the most earliest checkpoints.
        """
        with tf.variable_scope(self._scope + 'optimizer'):
            self.opt, _ = utils_tf.create_optimizer(optimizer, learning_rate,
                                                    optimizer_args=optimizer_args,
                                                    decay=decay, decay_args=decay_args,
                                                    global_step=self.global_step)
        save_vars = set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self._scope + 'network'))
        save_vars |= set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           self._scope + 'optimizer'))
        save_vars.add(self.global_step)
        save_vars.add(self._obs_counter)
        save_vars.add(self._ep_counter)
        self._saver = tf.train.Saver(var_list=list(save_vars), max_to_keep=saver_keep)
        # ASyNC add target
