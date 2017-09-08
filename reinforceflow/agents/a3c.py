from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time
import copy
from threading import Thread

from six.moves import range  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

import reinforceflow.utils
from reinforceflow.core.base_agent import BaseAgent
from reinforceflow.core import EGreedyPolicy, GreedyPolicy
from reinforceflow import utils_tf
from reinforceflow import logger
from reinforceflow.utils import discount_rewards
from reinforceflow.utils_tf import add_grads_summary, add_observation_summary
# TODO: Base async agent for async DQN and A3C
# TODO: Base Deep Agent
# TODO: Simplify and polish


class A3CAgent(BaseAgent):
    """Constructs Asynchronous Advantage Actor-Critic agent, based on paper:
    "Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016.
    (https://arxiv.org/abs/1602.01783v2)

    See `core.base_agent.BaseDQNAgent.__init__`.
    """
    def __init__(self, env, net_factory, use_gpu=False, name='A3C'):
        logger.warn("WARNING! A3C Agent is under development"
                    " and may contain some bugs in it's implementation.")
        super(A3CAgent, self).__init__(env=env)
        config = tf.ConfigProto(
            device_count={'GPU': use_gpu}
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self._net_factory = net_factory
        self._scope = '' if not name else name + '/'
        with tf.variable_scope(self._scope + 'network') as scope:
            self._action_ph = tf.placeholder('int32', [None] + self.env.action_shape, name='action')
            self._reward_ph = tf.placeholder('float32', [None], name='reward')
            self.net = self._net_factory.make(input_shape=[None] + self.env.obs_shape,
                                              output_size=self.env.action_shape[0])
            self._weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope.name)
        self._save_vars = set()
        self._no_op = tf.no_op()
        with tf.variable_scope(self._scope + 'optimizer'):
            self._no_op = tf.no_op()
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self._obs_counter = tf.Variable(0, trainable=False, name='obs_counter')
            self._obs_counter_inc = self._obs_counter.assign_add(1, use_locking=True)
        self.weights = self._weights
        self.request_stop = False
        self.opt = None
        self._lr = None
        self._saver = None
        self.writer = None
        self._init_op = None
        self._term_ph = None
        self._train_op = None
        self._last_time = None
        self._summary_op = None
        self._prev_obs_step = None
        self._prev_opt_step = None

    def _write_summary(self, test_episodes=3):
        test_r = self.test(episodes=test_episodes)
        avg_r = test_r.compute_average()
        max_r = test_r.max
        min_r = test_r.min
        obs_step = self.obs_counter
        obs_per_sec = (self.obs_counter - self._prev_obs_step) / (time.time() - self._last_time)
        opt_per_sec = (self.step_counter - self._prev_opt_step) / (time.time() - self._last_time)
        self._last_time = time.time()
        self._prev_obs_step = obs_step
        self._prev_opt_step = self.step_counter
        logger.info("Global agent greedy eval. Average R: %.2f. Step: %d."
                    % (avg_r, obs_step))
        logger.info("Performance. Observation/sec: %0.2f. Update/sec: %0.2f."
                    % (obs_per_sec, opt_per_sec))
        logs = [tf.Summary.Value(tag=self._scope + 'greedy_R', simple_value=avg_r),
                tf.Summary.Value(tag=self._scope + 'greedy_maxR', simple_value=max_r),
                tf.Summary.Value(tag=self._scope + 'greedy_minR', simple_value=min_r),
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
        with tf.variable_scope(self._scope + 'optimizer'):
            self.opt, self._lr = utils_tf.create_optimizer(optimizer, learning_rate,
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
              ignore_checkpoint=False,
              **kwargs):
        """Starts training of Asynchronous n-step Q-Learning agent.

        Args:
            num_threads: (int) Amount of asynchronous threads for training.
            steps: (int) Total amount of steps across all threads.
            optimizer: String or tensorflow Optimizer instance.
            learning_rate: (float) Optimizer learning rate.
            log_dir: (str) Directory used for summary and checkpoints.
            target_freq: (int) Target network update frequency (in update steps).
            log_freq: (int) Checkpoint and summary saving frequency (in update steps).
            optimizer_args: (dict) Keyword arguments used for optimizer creation.
            gradient_clip: (float) Norm gradient clipping. To disable, pass 0 or None.
            decay: (function) Learning rate decay.
                   Expects tensorflow decay function or function name string.
                   Available names: 'polynomial', 'exponential'.
                   To disable, pass None.
            decay_args: (dict) Keyword arguments used for learning rate decay function creation.
            policy: (core.BasePolicy) Agent's training policy.
            gamma: (float) Reward discount factor.
            batch_size: (int) Training batch size.
            render: (bool) Enables game screen rendering.
            saver_keep: (int) Maximum number of checkpoints can be stored in `log_dir`.
                        When exceeds, overwrites the most earliest checkpoints.
            ignore_checkpoint: (bool) If enabled, training will start from scratch,
                               and overwrite all old checkpoints found at `log_dir` path.
        """
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
                                      target_freq=target_freq,
                                      policy=policy[t],
                                      log_freq=log_freq,
                                      gradient_clip=gradient_clip,
                                      gamma=gamma,
                                      batch_size=batch_size,
                                      name='ThreadLearner%d' % t)
            thread_agents.append(agent)
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        if not ignore_checkpoint and tf.train.latest_checkpoint(log_dir) is not None:
            self.load_weights(log_dir)
        last_log_step = self.obs_counter

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
            except KeyboardInterrupt:
                logger.info('Caught Ctrl+C! Stopping training process.')
                self.request_stop = True
        self.save_weights(log_dir)
        logger.info('Training finished!')
        self.writer.close()
        for agent in thread_agents:
            agent.close()

    def _train_on_batch(self, obs, actions, rewards, obs_next, term, summarize=False):
        raise NotImplementedError('Training on batch is not supported. Use `train` method instead.')

    def train_on_batch(self, *args, **kwargs):
        return self._train_on_batch(*args, **kwargs)

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

    def increment_obs_counter(self):
        return self.sess.run(self._obs_counter_inc)

    @property
    def obs_counter(self):
        return self.sess.run(self._obs_counter_inc)

    @property
    def step_counter(self):
        return self.sess.run(self.global_step)

    def predict_action(self, obs, policy=GreedyPolicy()):
        """Computes action for given observation."""
        action_values = self.predict_on_batch([obs])
        return policy.select_action(self.env, action_values)

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


class _ThreadDQNLearner(BaseAgent, Thread):
    def __init__(self,
                 env,
                 net_factory,
                 global_agent,
                 steps,
                 target_freq,
                 policy,
                 log_freq,
                 gradient_clip=40.0,
                 gamma=0.99,
                 batch_size=32,
                 name=''):
        super(_ThreadDQNLearner, self).__init__(env=env)
        self._net_factory = net_factory
        self._scope = '' if not name else name + '/'
        self.sess = global_agent.sess
        self.global_agent = global_agent
        self._init_op = None
        self._save_vars = set()
        self.steps = steps
        self.target_freq = target_freq
        self.policy = policy
        self.log_freq = log_freq
        self.gamma = gamma
        self.batch_size = batch_size
        self._ep_reward = reinforceflow.utils.IncrementalAverage()
        self._ep_q = reinforceflow.utils.IncrementalAverage()
        self._reward_accum = 0

        # Inference Graph
        with tf.variable_scope(self._scope + 'network') as scope:
            self._action_ph = tf.placeholder('int32', [None] + self.env.action_shape, name='action')
            self._reward_ph = tf.placeholder('float32', [None], name='reward')
            self.net = self._net_factory.make(input_shape=[None] + self.env.obs_shape,
                                              output_size=self.env.action_shape[0])
            self._weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)

        # Train Graph
        with tf.variable_scope(self._scope + 'optimizer'):
            self._no_op = tf.no_op()
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self._obs_counter = tf.Variable(0, trainable=False, name='obs_counter')
            self._obs_counter_inc = self._obs_counter.assign_add(1, use_locking=True)
            action_argmax = tf.arg_max(self._action_ph, 1, name='action_argmax')
            self._action_onehot = tf.one_hot(action_argmax, self.env.action_shape[0],
                                             1.0, 0.0, name='action_one_hot')
            adv = self._reward_ph - self.net.output_value  # shape=B
            policy_logp = tf.log(self.net.output_policy + 1e-8)  # BxA
            loss_policy = -tf.reduce_sum(tf.reduce_sum(policy_logp * self._action_onehot, axis=1)
                                         * tf.stop_gradient(adv))  # shape=sum(B*B) -> 1
            loss_value = tf.reduce_sum(tf.square(adv))  # shape=sum(B) -> 1
            entropy = tf.reduce_sum(self.net.output_policy * policy_logp)  # shape=sum(BxA*BxA) -> 1
            self._loss = loss_policy + 0.5*loss_value + 0.01*entropy  # shape=1
            self._grads = tf.gradients(self._loss, self._weights)
            if gradient_clip:
                self._grads, _ = tf.clip_by_global_norm(self._grads, gradient_clip)
            self._grads_vars = list(zip(self._grads, self.global_agent.weights))
            self._train_op = self.global_agent.opt.apply_gradients(self._grads_vars,
                                                                   self.global_agent.global_step)
            self._sync_op = [self._weights[i].assign(self.global_agent.weights[i])
                             for i in range(len(self._weights))]
        add_grads_summary(self._grads_vars)
        with tf.variable_scope(self._scope):
            add_observation_summary(self.net.input_ph, self.env.obs_shape)
            tf.summary.histogram('output_policy', self.net.output)
            tf.summary.scalar('output_value', tf.reduce_mean(self.net.output_value))
            tf.summary.histogram('policy_log_prob', policy_logp)
            tf.summary.scalar('loss_policy', loss_policy)
            tf.summary.scalar('entropy', entropy)
            tf.summary.scalar('advantage', tf.reduce_mean(adv))
            tf.summary.scalar('loss_value', loss_value)
            tf.summary.scalar('loss', self._loss)
            self._summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                                  self._scope))

    def _sync_global(self):
        if self._sync_op is not None:
            self.sess.run(self._sync_op)

    def _train_on_batch(self, obs, actions, rewards, obs_next, term, summarize=False):
        expected_value = 0
        if not term:
            expected_value = self.sess.run(self.net.output_value, {self.net.input_ph: obs_next})
            self._ep_q.add(expected_value)
        else:
            self._ep_reward.add(self._reward_accum)
            self._reward_accum = 0
        rewards = discount_rewards(rewards, self.gamma, expected_value)
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
                reward_per_action = self.predict_on_batch([obs])
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
            summary_str = self._train_on_batch(batch_obs, batch_actions,
                                               batch_rewards, [obs], term, write_summary)
            if write_summary:
                prev_step = self.global_agent.obs_counter
                # TODO: wrap into separate func `add_reward_summary(reward)`
                num_ep = self._ep_reward.length
                max_r = self._ep_reward.max
                min_r = self._ep_reward.min
                avg_r = self._ep_reward.reset()
                avg_q = self._ep_q.reset()
                logger.info("%s on-policy eval: Average R: %.2f. Average maxQ: %.2f. Step: %d. "
                            % (self._scope, avg_r, avg_q, prev_step))
                if summary_str:
                    logs = [tf.Summary.Value(tag=self._scope + 'maxR', simple_value=max_r),
                            tf.Summary.Value(tag=self._scope + 'minR', simple_value=min_r),
                            tf.Summary.Value(tag=self._scope + 'avgR', simple_value=avg_r),
                            tf.Summary.Value(tag=self._scope + 'avgQ', simple_value=avg_q),
                            tf.Summary.Value(tag=self._scope + 'epsilon',
                                             simple_value=self.policy.epsilon),
                            tf.Summary.Value(tag=self._scope + 'metrics/num_episodes',
                                             simple_value=num_ep)
                            ]
                    self.global_agent.writer.add_summary(tf.Summary(value=logs),
                                                         global_step=prev_step)
                    self.global_agent.writer.add_summary(summary_str, global_step=prev_step)

    def predict_action(self, obs, policy=GreedyPolicy()):
        """Computes action for given observation."""
        action_values = self.predict_on_batch([obs])
        return policy.select_action(self.env, action_values)

    def predict_on_batch(self, obs_batch):
        """Computes action-values for given batch of observations."""
        return self.sess.run(self.net.output, {self.net.input_ph: obs_batch})

    def close(self):
        pass

    def train_on_batch(self, *args, **kwargs):
        raise NotImplementedError('Use `AsyncDQNAgent.train`.')

    def train(self, *args, **kwargs):
        raise NotImplementedError('Use `AsyncDQNAgent.train`.')
