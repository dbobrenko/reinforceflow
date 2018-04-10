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
from reinforceflow.core.stats import Stats, flush_stats


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
                self.actions = tf.placeholder('float32',
                                              [None] + list(self.env.action_space.shape),
                                              name='action')
                self.rewards = tf.placeholder('float32', [None], name='reward')
                self.net = self.model.build(input_space=self.env.observation_space,
                                            output_space=self.env.action_space)
                self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)
        # Train Part
        with tf.variable_scope(self._scope + 'optimizer'):
            self._no_op = tf.no_op()
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self._ep_counter = tf.Variable(0, trainable=False, name='ep_counter')
            self._obs_counter = tf.Variable(0, trainable=False, name='obs_counter')
        self._savings = set(self.weights)
        self._savings.add(self.global_step)
        self._savings.add(self._obs_counter)
        self._savings.add(self._ep_counter)
        self.lock = threading.Lock()
        self._saver = tf.train.Saver(self._savings)
        self.test_env = None

    def train_on_batch(self, obs, actions, rewards, obs_next, trajectory_ends,
                       term, lr, gamma=0.99, summarize=False):
        raise NotImplementedError

    def predict_on_batch(self, obs_batch):
        raise NotImplementedError

    def act(self, obs):
        """Computes greedy action for given observation.
        Args:
            obs (numpy.ndarray): Observation.
        Returns:
            Action.
        """
        raise NotImplementedError

    def explore(self, obs, step=None):
        """Computes action in exploration mode for given observation.
        Args:
            obs (numpy.ndarray): Observation.
            step (int): Current agent step. To use current step, pass None.
        Returns:
            Action.
        """
        raise NotImplementedError

    @property
    def name(self):
        return self._scope[:-1]

    @property
    def optimize_counter(self):
        return self.sess.run(self.opt.global_step)

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

    def test(self, env, episodes, max_steps=1e5, render=False, max_fps=None, writer=None):
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
        if env is not None:
            self.test_env = env
        elif self.test_env is None:
            logger.warn("Testing environment is not provided. Using training env as testing.")
            self.test_env = copy.deepcopy(self.env)
        stats = Stats(agent=self)
        delta_frame = 1. / max_fps if max_fps else 0
        step_counter = 0
        episode_counter = 0
        max_steps = int(max_steps)
        for _ in range(episodes):
            obs = self.test_env.reset()
            for i in range(max_steps):
                start_time = time.time()
                action = self.act(obs)
                obs, r, terminal, info = self.test_env.step(action)
                step_limit = i >= max_steps - 1
                terminal = terminal or step_limit
                if step_limit:
                    logger.info("Interrupting test episode due to the "
                                "maximum allowed number of steps (%d)" % i)
                step_counter += 1
                episode_counter += terminal
                stats.add(action, r, terminal, info)
                if render:
                    self.test_env.render()
                    if delta_frame > 0:
                        delay = max(0, delta_frame - (time.time() - start_time))
                        time.sleep(delay)
                if terminal:
                    # TODO: Check for atari life lost
                    break
        reward_stats = copy.deepcopy(stats.reward_stats)
        flush_stats(stats, log_progress=False, log_performance=False, log_hyperparams=False,
                    name='%s Test' % self.name, writer=writer)
        return reward_stats

    def async_test(self, env, num_episodes, render, max_steps=int(1e4)):
        if num_episodes <= 0:
            return

        with self.lock:
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
