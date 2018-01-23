from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import copy
import os
import threading
import time
from threading import Thread

import gym
import six
import tensorflow as tf
from six.moves import range  # pylint: disable=redefined-builtin

from reinforceflow import logger
from reinforceflow.core.policy import GreedyPolicy
from reinforceflow.core.space import Tuple, Continuous
from reinforceflow.core.stats import Stats


@six.add_metaclass(abc.ABCMeta)
class BaseAgent(object):
    @abc.abstractmethod
    def __init__(self, env, model, device='/gpu:0', name='', *args, **kwargs):
        """Abstract base class for Deep Network-based agents.

        Args:
            env (gym.Env): Environment instance.
            model (models.Model): Model builder.
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
        self.model = model
        self._scope = '' if not name else name + '/'
        self.device = device
        self.step = 0
        self.episode = 0
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
                self.net = self.model.build(input_space=self.env.observation_space,
                                            output_space=self.env.action_space)
                self._weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)
        # Train Part
        with tf.variable_scope(self._scope + 'optimizer'):
            self._no_op = tf.no_op()
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self._ep_counter = tf.Variable(0, trainable=False, name='ep_counter')
            self._obs_counter = tf.Variable(0, trainable=False, name='obs_counter')
        self._savings = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        self._savings |= set(self._weights)
        self._savings.add(self.global_step)
        self._savings.add(self._obs_counter)
        self._savings.add(self._ep_counter)
        self._mutex = threading.Lock()
        self._saver = None

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def train_on_batch(self, obs, actions, rewards, obs_next, term, lr,  summarize=False):
        raise NotImplementedError

    def predict_on_batch(self, obs_batch):
        """Computes action-values for given batch of observations."""
        return self.sess.run(self.net['out'], {self.net['in']: obs_batch})

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
        if isinstance(self.env.action_space, Continuous):
            return action_values
        if policy is None:
            policy = GreedyPolicy
        return policy.select_action(self.env, action_values, step)

    @property
    def name(self):
        return self._scope[:-1]

    def save_weights(self, path, model_name='model.ckpt'):
        if not os.path.exists(path):
            os.makedirs(path)
        self.sess.run(self._obs_counter.assign(self.step))
        self.sess.run(self._ep_counter.assign(self.episode))
        self._saver.save(self.sess, os.path.join(path, model_name), global_step=self.global_step)
        logger.info('Checkpoint has been saved to: %s' % os.path.join(path, model_name))

    def load_weights(self, checkpoint):
        if not os.path.exists(checkpoint):
            raise ValueError('Checkpoint path does not exists: %s' % checkpoint)
        if tf.gfile.IsDirectory(checkpoint):
            checkpoint = tf.train.latest_checkpoint(checkpoint)
        self._saver.restore(self.sess, save_path=checkpoint)
        self.step = self.sess.run(self._obs_counter)
        self.episode = self.sess.run(self._ep_counter)
        logger.info('Checkpoint has been restored from: %s' % checkpoint)

    def test(self, env, episodes, max_steps=int(1e5), render=False, max_fps=None, writer=None):
        """Tests agent's performance on a given number of episodes.

        Args:
            env (gym.Env): Test environment.
            episodes (int): Number of episodes.
            max_steps (int): Maximum allowed step per episode.
            render (bool): Enables game screen rendering.
            max_fps (int): Maximum allowed fps. To disable fps limitation, pass None.
            writer (FileWriter): TensorBoard summary writer.

        Returns (utils.RewardStats): Average reward per episode.
        """
        if env is None:
            logger.warn("Testing environment is not provided. Using training env as testing.")
            env = copy.deepcopy(self.env)
        stats = Stats(log_freq=None, log_on_term=False, log_prefix='Test', file_writer=writer,
                      log_performance=False)
        delta_frame = 1. / max_fps if max_fps else 0
        step_counter = 0
        episode_counter = 0
        for _ in range(episodes):
            obs = env.reset()
            for i in range(max_steps):
                start_time = time.time()
                action = self.predict_action(obs)
                obs, r, terminal, info = env.step(action)
                terminal = terminal or i >= max_steps - 1
                step_counter += 1
                episode_counter += terminal
                stats.add(r, terminal, info, step=step_counter, episode=episode_counter)
                stats.add(r, terminal, info, step=step_counter, episode=episode_counter)
                if render:
                    env.render()
                    if delta_frame > 0:
                        delay = max(0, delta_frame - (time.time() - start_time))
                        time.sleep(delay)
                if terminal:
                    # TODO: Check for atari life lost
                    break
        reward_stats = copy.deepcopy(stats.reward_stats)
        stats.flush(step=self.step, episode=self.episode)
        env.close()
        return reward_stats

    def async_test(self, env, num_episodes, render, max_steps=int(1e4)):
        if num_episodes <= 0:
            return

        with self._mutex:
            def evaluate():
                self.test(env=env, episodes=num_episodes, render=render, max_steps=max_steps)
            t = Thread(target=evaluate)
            t.daemon = True
            t.start()
            t.join()

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
class BaseDiscreteAgent(BaseAgent):
    """Base class for agents with discrete action space.
    See `BaseAgent`.
    """
    @abc.abstractmethod
    def __init__(self, env, model, device='/gpu:0', name='', **kwargs):
        super(BaseDiscreteAgent, self).__init__(env, model, device=device, name=name, **kwargs)
        if isinstance(self.env.action_space, Continuous):
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
class BaseDeepQ(BaseDiscreteAgent):
    """Base class for DQN-family agents with discrete action space.
    See `BaseDiscreteAgent`.

    Inherited fields:
        _target_net (models.AbstractFactory): Target network.
        _target_update (operation): TensorFlow Operation for target network update.
    """
    @abc.abstractmethod
    def __init__(self, env, model, device='/gpu:0', name='', **kwargs):
        super(BaseDeepQ, self).__init__(env, model, device=device, name=name, **kwargs)
        with tf.variable_scope(self._scope + 'target_network') as scope:
            self._target_net = self.model.build(input_space=self.env.observation_space,
                                                output_space=self.env.action_space)
            target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope.name)
            self._target_update = [target_weights[i].assign(self._weights[i])
                                   for i in range(len(target_weights))]

    def load_weights(self, checkpoint):
        super(BaseDeepQ, self).load_weights(checkpoint)
        self.target_update()

    def target_predict(self, obs):
        """Computes target network action-values with for given batch of observations."""
        return self.sess.run(self._target_net['out'], {self._target_net['in']: obs})

    def target_update(self):
        """Updates target network."""
        self.sess.run(self._target_update)
