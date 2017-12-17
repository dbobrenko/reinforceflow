from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import time
import threading
from threading import Thread

import six
import tensorflow as tf
from six.moves import range  # pylint: disable=redefined-builtin

import reinforceflow.utils
from reinforceflow import logger
from reinforceflow.core.policy import GreedyPolicy
from reinforceflow.core.space import Tuple, Continious


@six.add_metaclass(abc.ABCMeta)
class BaseAgent(object):
    @abc.abstractmethod
    def __init__(self, env, net_factory, device='/gpu:0', name='', *args, **kwargs):
        """Abstract base class for Deep Network-based agents.

        Args:
            env (gym.Env): Environment instance.
            net_factory (AbstractFactory): Network factory, defined in nets file.
            device (str): TensorFlow device.
            name (str): Agent's name prefix.

        Inherited fields:
            sess: TensorFlow session.
            _savings (set): Holds variables for Tensorflow Saver.
                During BaseAgent init fills with model weights, obs, episode and update counters.
            _scope (str): Agent's variable scope.


        Implement the following fields/methods:
            _saver (tensorflow.train.saver): Agent's saver/restorer field.
                Recommended to create with self._savings set (see "Inherited fields").
            train (method): Starts agent's training.

        Optionally implement:
            train_on_batch (method): Performs single network update on
                (obs, action, reward, next_obs, term) transition.

        """
        super(BaseAgent, self).__init__()
        self.env = env
        self._net_factory = net_factory
        self._scope = '' if not name else name + '/'
        self.device = device
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        with tf.device(self.device):
            # Inference Graph
            with tf.variable_scope(self._scope + 'network') as scope:
                self._action_ph = tf.placeholder('float32',
                                                 [None] + list(self.env.action_space.shape),
                                                 name='action')
                self._reward_ph = tf.placeholder('float32', [None], name='reward')
                self.net = self._net_factory.make(input_space=self.env.observation_space,
                                                  output_space=self.env.action_space)
                self._weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)
        # Train Part
        with tf.variable_scope(self._scope + 'optimizer'):
            self._no_op = tf.no_op()
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self._ep_counter = tf.Variable(0, trainable=False, name='ep_counter')
            self._ep_counter_inc = self._ep_counter.assign_add(1, use_locking=True)
            self._obs_counter = tf.Variable(0, trainable=False, name='obs_counter')
            self._obs_counter_inc = self._obs_counter.assign_add(1, use_locking=True)
        self._savings = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        self._savings |= set(self._weights)
        self._savings.add(self.global_step)
        self._savings.add(self._obs_counter)
        self._savings.add(self._ep_counter)
        self._saver = None
        self._mutex = threading.Lock()

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def train_on_batch(self, obs, actions, rewards, obs_next, term, summarize=False):
        raise NotImplementedError

    def predict_action(self, obs, policy=None, step=0):
        """Computes action for given observation.
        Args:
            obs (numpy.ndarray): Observation.
            policy (core.Policy): Policy, used for discrete action space agents.
                By default Greedy-policy is used.
            step (int): Observation counter, used for non-greedy policies.
        Returns:
            Raw network output, if environment action space is continuous.
            Action chosen by policy, if environment action space is discrete.
        """
        action_values = self.predict_on_batch([obs])[0]
        if isinstance(self.env.action_space, Continious):
            return action_values
        if policy is None:
            policy = GreedyPolicy
        return policy.select_action(self.env, action_values, step)

    @property
    def name(self):
        return self._scope[:-1]

    @property
    def ep_counter(self):
        return self.sess.run(self._ep_counter)

    @property
    def obs_counter(self):
        return self.sess.run(self._obs_counter)

    @property
    def step_counter(self):
        return self.sess.run(self.global_step)

    def increment_ep_counter(self):
        return self.sess.run(self._ep_counter_inc)

    def increment_obs_counter(self):
        return self.sess.run(self._obs_counter_inc)

    def save_weights(self, path, model_name='model.ckpt'):
        if not os.path.exists(path):
            os.makedirs(path)
        self._saver.save(self.sess, os.path.join(path, model_name), global_step=self.global_step)
        logger.info('Checkpoint has been saved to: %s' % os.path.join(path, model_name))

    def load_weights(self, checkpoint):
        if not os.path.exists(checkpoint):
            raise ValueError('Checkpoint path does not exists: %s' % checkpoint)
        if tf.gfile.IsDirectory(checkpoint):
            checkpoint = tf.train.latest_checkpoint(checkpoint)
        self._saver.restore(self.sess, save_path=checkpoint)
        logger.info('Checkpoint has been restored from: %s', checkpoint)

    def predict_on_batch(self, obs_batch):
        """Computes action-values for given batch of observations."""
        return self.sess.run(self.net.output, {self.net.input_ph: obs_batch})

    def test(self, episodes, render=False, max_fps=None, copy_env=False, max_steps=int(1e4)):
        """Tests agent's performance on a given number of episodes.

        Args:
            episodes (int): Number of episodes.
            render (bool): Enables game screen rendering.
            max_fps (int): Maximum allowed fps. To disable fps limitation, pass None.
            copy_env (bool): If enabled, performs tests on the copy of environment instance.
            max_steps (int): Maximum allowed steps per episode.

        Returns (utils.RewardStats): Average reward per episode.
        """
        # In seconds
        delta_frame = 1. / max_fps if max_fps else 0
        env = self.env.new() if copy_env else self.env
        rewards = reinforceflow.utils.RewardStats()
        for _ in range(episodes):
            obs = env.reset()
            for _ in range(max_steps):
                start_time = time.time()
                action = self.predict_action(obs)
                obs, r, terminal, info = env.step(action)
                rewards.add(r, terminal)
                if render:
                    env.render()
                    if delta_frame > 0:
                        delay = max(0, delta_frame - (time.time() - start_time))
                        time.sleep(delay)
                if terminal:
                    break
        return rewards

    def close(self):
        if self.sess:
            self.sess.close()

    def _async_eval(self, writer, reward_logger, num_episodes, render,
                    train_stats=None, log_fps=True):
        if num_episodes <= 0:
            return

        def evaluate():
            with self._mutex:
                test_stats = self.test(episodes=num_episodes, render=render, copy_env=True)
                reward_summary = reward_logger.summarize(train_stats,
                                                         test_stats,
                                                         self.ep_counter,
                                                         self.step_counter,
                                                         self.obs_counter,
                                                         scope=self._scope,
                                                         log_performance=log_fps)
                writer.add_summary(reward_summary, global_step=self.obs_counter)
        t = Thread(target=evaluate)
        t.daemon = True
        t.start()
        t.join()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@six.add_metaclass(abc.ABCMeta)
class BaseDiscreteAgent(BaseAgent):
    """Base class for agents with discrete action space.
    See `BaseAgent`.
    """
    @abc.abstractmethod
    def __init__(self, env, net_factory, device='/gpu:0', name='', **kwargs):
        super(BaseDiscreteAgent, self).__init__(env, net_factory, device=device,
                                                name=name, **kwargs)
        if isinstance(self.env.action_space, Continious):
            raise ValueError('%s does not support environments with continuous '
                             'action space.' % self.__class__.__name__)
        if isinstance(self.env.action_space, Tuple):
            raise ValueError('%s does not support environments with multiple '
                             'action spaces.' % self.__class__.__name__)

    def predict_action(self, obs, policy=GreedyPolicy, step=0):
        """Computes action with given policy for given observation.
        See `Agent.predict_action`."""
        action_values = self.predict_on_batch([obs])
        return policy.select_action(self.env, action_values, step=step)


@six.add_metaclass(abc.ABCMeta)
class BaseDQNAgent(BaseDiscreteAgent):
    """Base class for DQN-family agents with discrete action space.
    See `BaseDiscreteAgent`.

    Inherited fields:
        _target_net (nets.AbstractFactory): Target network.
        _target_update (operation): TensorFlow Operation for target network update.
    """
    @abc.abstractmethod
    def __init__(self, env, net_factory, device='/gpu:0', name='', **kwargs):
        super(BaseDQNAgent, self).__init__(env, net_factory, device=device, name=name, **kwargs)
        with tf.variable_scope(self._scope + 'target_network') as scope:
            self._target_net = self._net_factory.make(input_space=self.env.observation_space,
                                                      output_space=self.env.action_space)
            target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope.name)
            self._target_update = [target_weights[i].assign(self._weights[i])
                                   for i in range(len(target_weights))]

    def load_weights(self, checkpoint):
        super(BaseDQNAgent, self).load_weights(checkpoint)
        self.target_update()

    def target_predict(self, obs):
        """Computes target network action-values with for given batch of observations."""
        return self.sess.run(self._target_net.output, {self._target_net.input_ph: obs})

    def target_update(self):
        """Updates target network."""
        self.sess.run(self._target_update)
