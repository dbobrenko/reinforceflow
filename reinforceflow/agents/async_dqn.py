from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import random
import copy
from threading import Thread

from six.moves import range  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

from reinforceflow.core.base_agent import BaseDQNAgent
from reinforceflow.core import EGreedyPolicy
from reinforceflow import misc
from reinforceflow import logger
from reinforceflow.misc import discount_rewards


class AsyncDQNAgent(BaseDQNAgent):
    """Constructs Asynchronous N-step Q-Learning agent, based on paper:
    "Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2015.
    (https://arxiv.org/abs/1602.01783v2)

    See `core.base_agent.BaseDQNAgent.__init__`.
    """
    def __init__(self, env, net_factory, use_gpu=False, name='AsyncDQN'):
        super(AsyncDQNAgent, self).__init__(env=env, net_factory=net_factory, name=name)
        config = tf.ConfigProto(
            device_count={'GPU': use_gpu}
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.weights = self._weights
        self.request_stop = False
        self._prev_obs_step = None
        self._prev_opt_step = None
        self._last_time = None
        self.writer = None
        self.sess.run(tf.global_variables_initializer())

    def _write_summary(self, test_episodes=3):
        test_r, test_q = self.test(episodes=test_episodes)
        obs_step = self.obs_counter
        obs_per_sec = (self.obs_counter - self._prev_obs_step) / (time.time() - self._last_time)
        opt_per_sec = (self.step_counter - self._prev_opt_step) / (time.time() - self._last_time)
        self._last_time = time.time()
        self._prev_obs_step = obs_step
        self._prev_opt_step = self.step_counter
        logger.info("Global agent greedy eval: Average R: %.2f. Average maxQ: %.2f. Step: %d."
                    % (test_r, test_q, obs_step))
        logger.info("Performance. Observation/sec: %0.2f. Update/sec: %0.2f."
                    % (obs_per_sec, opt_per_sec))
        logs = [tf.Summary.Value(tag=self._scope + 'greedy_r', simple_value=test_r),
                tf.Summary.Value(tag=self._scope + 'greedy_q', simple_value=test_q),
                tf.Summary.Value(tag='performance/observation/sec', simple_value=obs_per_sec),
                tf.Summary.Value(tag='performance/update/sec', simple_value=opt_per_sec)
                ]
        self.writer.add_summary(tf.Summary(value=logs), global_step=obs_step)

    def build_train_graph(self, optimizer, learning_rate, optimizer_args=None,
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
            self._target_net =\
                self._net_factory.make(input_shape=[None] + self.env.observation_shape,
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
        self._save_vars |= set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 self._scope + 'network'))
        self._save_vars |= set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 self._scope + 'optimizer'))
        self._save_vars.add(self.global_step)
        self._save_vars.add(self._obs_counter)
        self._saver = tf.train.Saver(var_list=list(self._save_vars), max_to_keep=saver_keep)
        self._summary_op = tf.no_op()

    def train(self,
              num_threads,
              steps,
              optimizer,
              learning_rate,
              log_dir,
              target_freq,
              log_freq,
              optimizer_args=None,
              gradient_clip=40.0,
              decay=None,
              decay_args=None,
              policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=20000),
              gamma=0.99,
              batch_size=32,
              render=False,
              saver_keep=10,
              **kwargs):
        if num_threads < 1:
            raise ValueError("Number of threads must be >= 1 (Got: %s)." % num_threads)
        thread_agents = []
        envs = []

        if isinstance(policy, (list, tuple, np.ndarray)):
            if len(policy) != num_threads:
                raise ValueError("Amount of policies should be equal to the amount of threads.")
        else:
            policy = [copy.deepcopy(policy) for _ in range(num_threads)]

        self.build_train_graph(optimizer, learning_rate, optimizer_args=optimizer_args,
                               decay=decay, decay_args=decay_args,
                               gradient_clip=gradient_clip, saver_keep=saver_keep)
        for t in range(num_threads):
            env = self.env.copy()
            envs.append(env)
            agent = _ThreadDQNLearner(env=env,
                                      net_factory=self._net_factory,
                                      global_agent=self,
                                      steps=steps,
                                      optimizer=optimizer,
                                      learning_rate=learning_rate,
                                      target_freq=target_freq,
                                      policy=policy[t],
                                      log_freq=log_freq,
                                      optimizer_args=optimizer_args,
                                      decay=decay,
                                      decay_args=decay_args,
                                      gradient_clip=gradient_clip,
                                      gamma=gamma,
                                      batch_size=batch_size,
                                      saver_keep=saver_keep,
                                      name='ThreadLearner%d' % t)
            thread_agents.append(agent)
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        if log_dir and tf.train.latest_checkpoint(log_dir) is not None:
            self.load_weights(log_dir)
        last_log_step = self.obs_counter
        last_target_update = last_log_step

        for t in thread_agents:
            t.daemon = True
            t.start()
        self.request_stop = False

        def has_live_threads():
            return True in [th.isAlive() for th in thread_agents]

        self._prev_obs_step = self.obs_counter
        self._prev_opt_step = self.step_counter
        self._last_time = time.time()
        while has_live_threads() and self.obs_counter < steps:
            try:
                if render:
                    for env in envs:
                        env.render()
                    time.sleep(0.01)
                step = self.obs_counter
                if step - last_log_step >= log_freq:
                    last_log_step = step
                    self._write_summary()
                    self.save_weights(log_dir)
                if step - last_target_update >= target_freq:
                    last_target_update = step
                    self.target_update()
            except KeyboardInterrupt:
                logger.info('Caught Ctrl+C! Stopping training process.')
                self.request_stop = True
        self.save_weights(log_dir)
        logger.info('Training finished!')
        self.writer.close()
        for agent in thread_agents:
            agent.close()

    def train_on_batch(self, obs, actions, rewards, obs_next, term, summarize=False):
        raise NotImplementedError('Training on batch is not supported. Use `train` method instead.')


class _ThreadDQNLearner(BaseDQNAgent, Thread):
    def __init__(self,
                 env,
                 net_factory,
                 global_agent,
                 steps,
                 optimizer,
                 learning_rate,
                 target_freq,
                 policy,
                 log_freq,
                 optimizer_args=None,
                 decay=None,
                 decay_args=None,
                 gradient_clip=40.0,
                 gamma=0.99,
                 batch_size=32,
                 saver_keep=10,
                 name=''):
        super(_ThreadDQNLearner, self).__init__(env=env, net_factory=net_factory, name=name)
        self.global_agent = global_agent
        self.sess = global_agent.sess
        self._sync_op = None
        self.build_train_graph(optimizer, learning_rate, optimizer_args, decay, decay_args,
                               gradient_clip, saver_keep)
        self.steps = steps
        self.target_freq = target_freq
        self.policy = policy
        self.log_freq = log_freq
        self.gamma = gamma
        self.batch_size = batch_size
        self._ep_reward = misc.IncrementalAverage()
        self._ep_q = misc.IncrementalAverage()
        self._reward_accum = 0

    def build_train_graph(self, optimizer, learning_rate, optimizer_args=None,
                          decay=None, decay_args=None, gradient_clip=40.0, saver_keep=10):
        # TODO: fix Variable already exists bug while creating the 2nd agent in the same scope
        with tf.variable_scope(self._scope + 'optimizer'):
            self._action_onehot = tf.one_hot(self._action_ph, self.env.action_shape, 1.0, 0.0,
                                             name='action_one_hot')
            q_selected = tf.reduce_sum(self.net.inference_op * self._action_onehot, 1)
            td_error = self._reward_ph - q_selected
            self._loss = tf.reduce_mean(tf.square(td_error), name='loss')
            self._grads = tf.gradients(self._loss, self._weights)
            if gradient_clip:
                self._grads, _ = tf.clip_by_global_norm(self._grads, gradient_clip)
            self._grads_vars = list(zip(self._grads, self.global_agent.weights))
            self._train_op = self.global_agent.opt.apply_gradients(self._grads_vars,
                                                                   self.global_agent.global_step)
            self._sync_op = [self._weights[i].assign(self.global_agent.weights[i])
                             for i in range(len(self._weights))]
        for grad, w in self._grads_vars:
            tf.summary.histogram(w.name, w)
            tf.summary.histogram(w.name + '/gradients', grad)
        with tf.variable_scope(self._scope):
            if len(self.env.observation_shape) == 1:
                tf.summary.histogram('observation', self.net.input_ph)
            elif len(self.env.observation_shape) <= 3:
                tf.summary.image('observation', self.net.input_ph)
            else:
                logger.warn('Cannot create summary for observation with shape %s'
                            % self.env.obs_shape)
            tf.summary.histogram('action', self._action_onehot)
            tf.summary.histogram('reward_per_action', self.net.inference_op)
            tf.summary.scalar('loss', self._loss)
            self._summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                                  self._scope))

    def _sync_global(self):
        if self._sync_op is not None:
            self.sess.run(self._sync_op)

    def _train_on_batch(self, obs, actions, rewards, obs_next, term, summarize=False):
        expected_reward = 0
        if not term:
            expected_reward = np.max(self.global_agent.target_predict(obs_next))
            self._ep_q.add(expected_reward)
        else:
            self._ep_reward.add(self._reward_accum)
            self._reward_accum = 0
        rewards = discount_rewards(rewards, self.gamma, expected_reward)
        _, summary = self.sess.run([self._train_op, self._summary_op if summarize else self._no_op],
                                   feed_dict={
                                       self.net.input_ph: obs,
                                       self._action_ph: actions,
                                       self._reward_ph: rewards
                                   })
        return summary

    def run(self):
        self._ep_reward.reset()
        self._ep_q.reset()
        self._reward_accum = 0
        prev_step = self.global_agent.obs_counter
        obs = self.env.reset()
        term = True
        while not self.global_agent.request_stop:
            self._sync_global()
            batch_obs, batch_rewards, batch_actions = [], [], []
            if term:
                term = False
                obs = self.env.reset()
            while not term and len(batch_obs) < self.batch_size:
                current_step = self.global_agent.increment_obs_counter()
                reward_per_action = self.predict(obs)
                batch_obs.append(obs)
                action = self.policy.select_action(self.env, reward_per_action, current_step)
                obs, reward, term, info = self.env.step(action)
                self._reward_accum += reward
                reward = np.clip(reward, -1, 1)
                batch_rewards.append(reward)
                batch_actions.append(action)
            write_summary = (term
                             and self.log_freq
                             and self.global_agent.obs_counter - prev_step > self.log_freq)
            summary_str = self._train_on_batch(np.vstack(batch_obs), batch_actions,
                                               batch_rewards, obs, term, write_summary)
            if write_summary:
                prev_step = self.global_agent.obs_counter
                train_r = self._ep_reward.reset()
                train_q = self._ep_q.reset()
                logger.info("%s on-policy eval: Average R: %.2f. Average maxQ: %.2f. Step: %d. "
                            % (self._scope, train_r, train_q, prev_step))
                if summary_str:
                    logs = [tf.Summary.Value(tag=self._scope + 'train_r', simple_value=train_r),
                            tf.Summary.Value(tag=self._scope + 'train_q', simple_value=train_q),
                            tf.Summary.Value(tag=self._scope + 'epsilon',
                                             simple_value=self.policy.epsilon)
                            ]
                    self.global_agent.writer.add_summary(tf.Summary(value=logs),
                                                         global_step=prev_step)
                    self.global_agent.writer.add_summary(summary_str, global_step=prev_step)

    def close(self):
        pass

    def train_on_batch(self, *args, **kwargs):
        raise NotImplementedError('Use `AsyncDQNAgent.train`.')

    def train(self, *args, **kwargs):
        raise NotImplementedError('Use `AsyncDQNAgent.train`.')
