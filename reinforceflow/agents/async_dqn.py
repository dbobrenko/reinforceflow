from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import copy
from threading import Thread

from six.moves import range  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

import reinforceflow.utils
from reinforceflow.core.base_agent import BaseDQNAgent
from reinforceflow.core import EGreedyPolicy
from reinforceflow import utils_tf
from reinforceflow import logger
from reinforceflow.utils import discount_rewards
from reinforceflow.utils_tf import add_grads_summary, add_observation_summary


class AsyncDQNAgent(BaseDQNAgent):
    """Constructs Asynchronous N-step Q-Learning agent, based on paper:
    "Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016.
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
        with tf.variable_scope(self._scope + 'target_network') as scope:
            self._target_net = self._net_factory.make(input_space=self.env.obs_space,
                                                      output_space=self.env.action_space)
            target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope.name)
            self._target_update = [target_weights[i].assign(self._weights[i])
                                   for i in range(len(target_weights))]

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

    def train(self,
              num_threads,
              steps,
              optimizer,
              learning_rate,
              batch_size,
              policy,
              target_freq,
              log_dir,
              optimizer_args=None,
              gradient_clip=40.0,
              decay=None,
              decay_args=None,
              gamma=0.99,
              log_every_sec=180,
              render=False,
              saver_keep=3,
              ignore_checkpoint=False,
              test_render=False,
              test_episodes=3,
              **kwargs):
        """Starts training of Asynchronous n-step Q-Learning agent.

        Args:
            num_threads: (int) Amount of asynchronous threads for training.
            steps: (int) Total amount of steps across all threads.
            optimizer: String or tensorflow Optimizer instance.
            learning_rate: (float) Optimizer learning rate.
            optimizer_args: (dict) Keyword arguments used for optimizer creation.
            gradient_clip: (float) Norm gradient clipping. To disable, pass 0 or None.
            batch_size: (int) Training batch size.
            policy: (core.BasePolicy) Agent's training policy.
            target_freq: (int) Target network update frequency (in update steps).
            log_dir: (str) Directory path, used for summary and checkpoints.
            decay: (function) Learning rate decay.
                   Expects tensorflow decay function or function name string.
                   Available names: 'polynomial', 'exponential'.
                   To disable, pass None.
            decay_args: (dict) Keyword arguments used for learning rate decay function creation.
            gamma: (float) Reward discount factor.
            log_every_sec: (int) Checkpoint and summary saving frequency (in seconds).
            render: (bool) Enables game screen rendering.
            saver_keep: (int) Maximum number of checkpoints can be stored in `log_dir`.
                        When exceeds, overwrites the most earliest checkpoints.
            ignore_checkpoint: (bool) If enabled, training will start from scratch,
                               and overwrite all old checkpoints found at `log_dir` path.
            test_render: (bool) Enables rendering for test evaluations.
            test_episodes: (int) Number of test episodes. To disable test evaluation, pass 0.
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
            agent = _ThreadDQNAgent(env=env,
                                    net_factory=self._net_factory,
                                    global_agent=self,
                                    policy=policy[t],
                                    log_every_sec=log_every_sec,
                                    gradient_clip=gradient_clip,
                                    gamma=gamma,
                                    batch_size=batch_size,
                                    name='ThreadAgent%d' % t)
            thread_agents.append(agent)
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        if not ignore_checkpoint and tf.train.latest_checkpoint(log_dir) is not None:
            self.load_weights(log_dir)
        last_log_time = time.time()
        last_target_update = self.obs_counter
        reward_logger = utils_tf.SummaryLogger(self.step_counter, self.obs_counter)

        for t in thread_agents:
            t.daemon = True
            t.start()
        self.request_stop = False

        def has_live_threads():
            return True in [th.isAlive() for th in thread_agents]

        def save_and_log():
            test_rewards = self.test(episodes=test_episodes, render=test_render)
            reward_summary = reward_logger.summarize(None, test_rewards,
                                                     self.ep_counter,
                                                     self.step_counter,
                                                     self.obs_counter,
                                                     scope=self._scope)
            self.writer.add_summary(reward_summary, global_step=self.obs_counter)
            self.save_weights(log_dir)

        while has_live_threads() and self.obs_counter < steps:
            try:
                if render:
                    for env in envs:
                        env.render()
                    time.sleep(0.01)
                if time.time() - last_log_time >= log_every_sec:
                    last_log_time = time.time()
                    save_and_log()
                if self.obs_counter - last_target_update >= target_freq:
                    last_target_update = self.obs_counter
                    self.target_update()
            except KeyboardInterrupt:
                logger.info('Caught Ctrl+C! Stopping training process.')
        self.request_stop = True
        logger.info('\nFinal evaluation:')
        save_and_log()
        self.save_weights(log_dir)
        logger.info('Training finished!')
        self.writer.close()
        for agent in thread_agents:
            agent.close()

    def train_on_batch(self, obs, actions, rewards, obs_next, term, summarize=False):
        raise NotImplementedError('Training on batch is not supported. Use `train` method instead.')


class _ThreadDQNAgent(BaseDQNAgent, Thread):
    def __init__(self,
                 env,
                 net_factory,
                 global_agent,
                 policy,
                 batch_size,
                 gradient_clip,
                 gamma,
                 log_every_sec,
                 name=''):
        super(_ThreadDQNAgent, self).__init__(env=env, net_factory=net_factory, name=name)
        self.global_agent = global_agent
        self.sess = global_agent.sess
        self.policy = policy
        self.batch_size = batch_size
        self.gamma = gamma
        self.log_every_sec = log_every_sec
        # Build train graph
        with tf.variable_scope(self._scope + 'optimizer'):
            action_argmax = tf.arg_max(self._action_ph, 1, name='action_argmax')
            action_onehot = tf.one_hot(action_argmax, self.env.action_space.shape[0],
                                       1.0, 0.0, name='action_one_hot')
            q_selected = tf.reduce_sum(self.net.output * action_onehot, 1)
            td_error = self._reward_ph - q_selected
            loss = tf.reduce_mean(tf.square(td_error), name='loss')
            grads = tf.gradients(loss, self._weights)
            if gradient_clip:
                grads, _ = tf.clip_by_global_norm(grads, gradient_clip)
            grads_vars = list(zip(grads, self.global_agent.weights))
            self._train_op = self.global_agent.opt.apply_gradients(grads_vars,
                                                                   self.global_agent.global_step)
            self._sync_op = [self._weights[i].assign(self.global_agent.weights[i])
                             for i in range(len(self._weights))]
        add_grads_summary(grads_vars)
        with tf.variable_scope(self._scope):
            add_observation_summary(self.net.input_ph, self.env.obs_space.shape)
            tf.summary.histogram('action', action_onehot)
            tf.summary.histogram('action_values', self.net.output)
            tf.summary.scalar('loss', loss)
            self._summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                                  self._scope))
        self._ep_reward = reinforceflow.utils.IncrementalAverage()
        self._ep_q = reinforceflow.utils.IncrementalAverage()
        self._reward_accum = 0

    def _sync_global(self):
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
        reward_logger = utils_tf.SummaryLogger(self.global_agent.step_counter,
                                               self.global_agent.obs_counter)
        self._ep_reward.reset()
        self._ep_q.reset()
        self._reward_accum = 0
        last_log_time = self.global_agent.obs_counter
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
                current_step = self.global_agent.increment_obs_counter()
                batch_obs.append(obs)
                reward_per_action = self.predict_on_batch([obs])
                action = self.policy.select_action(self.env, reward_per_action, current_step)
                obs, reward, term, info = self.env.step(action)
                self._reward_accum += reward
                reward = np.clip(reward, -1, 1)
                batch_rewards.append(reward)
                batch_actions.append(action)
            write_summary = (term and self.log_every_sec
                             and time.time() - last_log_time > self.log_every_sec)
            summary_str = self._train_on_batch(batch_obs, batch_actions,
                                               batch_rewards, [obs], term, write_summary)
            if write_summary:
                last_log_time = time.time()
                obs_step = self.global_agent.obs_counter
                reward_summary = reward_logger.summarize(self._ep_reward, None,
                                                         self.global_agent.ep_counter,
                                                         self.global_agent.step_counter,
                                                         self.global_agent.obs_counter,
                                                         log_performance=False,
                                                         scope=self._scope)
                self.global_agent.writer.add_summary(reward_summary, global_step=obs_step)
                avg_q = self._ep_q.reset()
                logs = [tf.Summary.Value(tag=self._scope + 'avg_Q', simple_value=avg_q),
                        tf.Summary.Value(tag=self._scope + 'epsilon',
                                         simple_value=self.policy.epsilon)]
                self.global_agent.writer.add_summary(tf.Summary(value=logs), global_step=obs_step)
                if summary_str:
                    self.global_agent.writer.add_summary(summary_str, global_step=obs_step)

    def close(self):
        pass

    def train_on_batch(self, *args, **kwargs):
        raise NotImplementedError('Use `AsyncDQNAgent.train`.')

    def train(self, *args, **kwargs):
        raise NotImplementedError('Use `AsyncDQNAgent.train`.')

    def build_train_graph(self, *args, **kwargs):
        raise NotImplementedError
