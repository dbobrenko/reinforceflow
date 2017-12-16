from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time
from threading import Thread

import numpy as np
import tensorflow as tf
from six.moves import range  # pylint: disable=redefined-builtin

import reinforceflow.utils
from reinforceflow import logger
from reinforceflow.core import BasePolicy
from reinforceflow.core.agent import BaseAgent
from reinforceflow.core.optimizer import Optimizer
from reinforceflow.utils import tensor_utils


class AsyncAgent(BaseAgent):
    def __init__(self, env, net_factory, thread_agent, optimizer, steps, policy, num_threads,
                 batch_size, device='/gpu:0', gamma=0.99, saver_keep=3,
                 log_every_sec=300, name=''):
        """Base class for asynchronous agents, based on paper:
        "Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016.
        (https://arxiv.org/abs/1602.01783v2)

        See `BaseAgent`.

        Args:
            env (gym.Env): Environment instance.
            net_factory (nets.AbstractFactory): Network factory.
            steps (int): [Training-only] Total amount of seen observations across all threads.
            optimizer (str or Optimizer): [Training-only] Agent's optimizer.
                By default: RMSProp(lr=4e-7, epsilon=0.1, decay=0.99, lrdecay='linear').
            policy (core.BasePolicy): [Training-only] Agent's training policy.
            num_threads (int): [Training-only] Amount of asynchronous threads for training.
            batch_size (int): [Training-only] Training batch size.
            device (str): TensorFlow device, used for graph creation.
            gamma (float): [Training-only] Reward discount factor.
            saver_keep (int): [Training-only] Maximum number of checkpoints can be stored at once.
            log_every_sec (int): [Training-only] Checkpoint and summary saving frequency
                (in seconds).
        """
        super(AsyncAgent, self).__init__(env=env, net_factory=net_factory, name=name)
        self._steps = steps
        self._log_freq = log_every_sec
        # Train Graph
        with tf.device(self.device):
            with tf.variable_scope(self._scope + 'optimizer') as sc:
                self.opt = Optimizer.create(optimizer)
                self.opt.build(steps, self.global_step, self._obs_counter)
                self._savings |= set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, sc.name))
                with tf.device('/cpu:0'):
                    tf.summary.scalar('lr', self.opt.lr)
        self._saver = tf.train.Saver(var_list=list(self._savings), max_to_keep=saver_keep)
        self.weights = self._weights
        if isinstance(policy, BasePolicy):
            policy = [copy.deepcopy(policy) for _ in range(num_threads)]
        if len(policy) != num_threads:
            raise ValueError("Amount of policies must be equal to the amount of threads.")
        self.writer = None
        self.request_stop = False
        if num_threads < 1:
            raise ValueError("Number of threads must be >= 1 (Got: %s)." % num_threads)
        self._thread_agents = []
        for t in range(num_threads):
            agent = thread_agent(env=self.env.new(),
                                 net_factory=self._net_factory,
                                 global_agent=self,
                                 policy=policy[t],
                                 batch_size=batch_size,
                                 gamma=gamma,
                                 log_every_sec=self._log_freq,
                                 device=device,
                                 name='ThreadAgent%d' % t)
            self._thread_agents.append(agent)

    def train(self, log_dir, render=False, test_render=False, test_episodes=1, callbacks=set()):
        """Starts training.

        Args:
            log_dir (str): Path used for summary and checkpoints.
            render (bool): Enables game screen rendering.
            test_render (bool): Enables rendering for test evaluations.
            test_episodes (int): Number of test episodes. To disable test evaluation, pass 0.
            callbacks (set): Set of AgentCallback instances.
        """
        last_log_time = time.time()
        reward_logger = tensor_utils.SummaryLogger(self.step_counter, self.obs_counter)

        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        for t in self._thread_agents:
            t.daemon = True
            t.start()
        self.request_stop = False
        try:
            while True:
                obs_step = self.obs_counter
                logs = {'obs_counter': obs_step}
                [callback.on_iter_start(self, logs) for callback in callbacks]
                if obs_step > self._steps:
                    break
                if time.time() - last_log_time >= self._log_freq:
                    last_log_time = time.time()
                    self.save_weights(log_dir)
                    self._async_eval(self.writer, reward_logger, test_episodes, test_render)
                    [callback.on_log(self, logs) for callback in callbacks]
                if render:
                    [agent.env.render() for agent in self._thread_agents]
                [callback.on_iter_end(self, logs) for callback in callbacks]
        except KeyboardInterrupt:
            logger.info('Caught Ctrl+C! Stopping training process.')
        self.request_stop = True
        logger.info('Saving progress & performing evaluation.')
        self.save_weights(log_dir)
        self._async_eval(self.writer, reward_logger, test_episodes, test_render)
        [t.join() for t in self._thread_agents]
        logger.info('Training finished!')
        self.writer.close()

    def train_on_batch(self, *args, **kwargs):
        raise ValueError("Training on batch is not supported. Use `train` method instead.")


class AsyncThreadAgent(BaseAgent, Thread):
    def __init__(self, env, net_factory, global_agent, policy, gamma,
                 batch_size, device, log_every_sec, name=''):
        super(AsyncThreadAgent, self).__init__(env=env, net_factory=net_factory, name=name)
        self.sess = global_agent.sess
        self.global_agent = global_agent
        self.policy = policy
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.log_every_sec = log_every_sec
        self._ep_reward = reinforceflow.utils.IncrementalAverage()
        self._ep_q = reinforceflow.utils.IncrementalAverage()
        self._reward_accum = 0

    def train_on_batch(self, obs, actions, rewards, obs_next, term, summarize=False):
        raise NotImplementedError

    def _sync_global(self):
        self.sess.run(self._sync_op)

    def run(self):
        reward_logger = tensor_utils.SummaryLogger(self.global_agent.step_counter,
                                                   self.global_agent.obs_counter)
        self._ep_reward.reset()
        self._ep_q.reset()
        self._reward_accum = 0
        last_log_time = time.time()
        obs = self.env.reset()
        term = True
        while not self.global_agent.request_stop:
            self._sync_global()
            batch_obs, batch_rewards, batch_actions = [], [], []
            if term:
                term = False
                obs = self.env.reset()
                self.global_agent.increment_ep_counter()
            while not term and len(batch_obs) < self.batch_size:
                obs_counter = self.global_agent.increment_obs_counter()
                batch_obs.append(obs)
                action = self.predict_action(obs, self.policy, obs_counter)
                obs, reward, term, info = self.env.step(action)

                self._reward_accum += reward
                reward = np.clip(reward, -1, 1)
                batch_rewards.append(reward)
                batch_actions.append(action)
            write_summary = (term and self.log_every_sec
                             and time.time() - last_log_time > self.log_every_sec)
            summary_str = self.train_on_batch(batch_obs, batch_actions, batch_rewards, [obs],
                                              term, write_summary)
            if write_summary:
                last_log_time = time.time()
                reward_summary = reward_logger.summarize(self._ep_reward, None,
                                                         self.global_agent.ep_counter,
                                                         self.global_agent.step_counter,
                                                         obs_counter,
                                                         q_values=self._ep_q,
                                                         log_performance=False,
                                                         scope=self._scope)
                self.global_agent.writer.add_summary(reward_summary, global_step=obs_counter)
                avg_q = self._ep_q.reset()
                logs = [tf.Summary.Value(tag=self._scope + 'avg_Q', simple_value=avg_q),
                        tf.Summary.Value(tag=self._scope + 'epsilon',
                                         simple_value=self.policy.epsilon)]
                self.global_agent.writer.add_summary(tf.Summary(value=logs),
                                                     global_step=obs_counter)
                if summary_str:
                    self.global_agent.writer.add_summary(summary_str, global_step=obs_counter)

    def close(self):
        pass

    def train(self, *args, **kwargs):
        raise ValueError('Use `A3CAgent.train` instead.')
