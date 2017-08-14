from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

from six.moves import range  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

from reinforceflow.core.base_agent import BaseDQNAgent
from reinforceflow.core import ExperienceReplay
from reinforceflow.nets import dqn
from reinforceflow.core import EGreedyPolicy
from reinforceflow import misc
from reinforceflow import logger


class DQNAgent(BaseDQNAgent):
    def __init__(self, env, net_fn=dqn, use_double=True, use_gpu=True, name=''):
        """Constructs Deep Q-Network agent, based on paper:
        "Human-level control through deep reinforcement learning", Mnih et al., 2015.

        See `core.base_agent.BaseDQNAgent.__init__`.
        Args:
            use_double: (bool) Enables Double DQN.
        """
        super(DQNAgent, self).__init__(env=env, net_fn=net_fn, name=name)
        config = tf.ConfigProto(
            device_count={'GPU': use_gpu}
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self._build_inference_graph(self.env)
        self.sess.run(tf.global_variables_initializer())
        self._use_double = use_double
        self._importance_ph = tf.placeholder('float32', [None], name='importance_sampling')
        self._td_error = None

    def build_train_graph(self, optimizer, learning_rate, optimizer_args=None, gamma=0.99,
                          decay=None, decay_args=None, gradient_clip=40.0, saver_keep=10):
        """Builds training graph.

        Args:
            optimizer: An optimizer name string or class.
            learning_rate (float or Tensor): Optimizer's learning rate.
            optimizer_args (dict): Keyword arguments used for optimizer creation.
            gamma: (float) Reward discount factor.
            decay (function): Learning rate decay.
                              Expects tensorflow decay function or function name string.
                              Available name strings: 'polynomial', 'exponential'.
                              To disable, pass None.
            decay_args (dict): Keyword arguments, passed to the decay function.
            gradient_clip (float): Norm gradient clipping.
                                   To disable, pass False or None.
            saver_keep (int): Maximum number of checkpoints can be stored in `log_dir`.
                              When exceeds, overwrites the most earliest checkpoints.
            replay: Experience replay
        """
        if self._train_op is not None:
            logger.warn("The training graph has already been built. Skipping.")
            return
        self._term_ph = tf.placeholder('float32', [None], name='term')
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
            self._action_onehot = tf.one_hot(self._action_ph, self.env.action_shape, 1.0, 0.0,
                                             name='action_one_hot')
            # Predict expected future reward for performed action
            q_selected = tf.reduce_sum(self._q * self._action_onehot, 1)
            if self._use_double:
                q_next_online_argmax = tf.arg_max(self._q, 1)
                q_next_online_onehot = tf.one_hot(q_next_online_argmax, self.env.action_shape, 1.0)
                q_next_max = tf.reduce_sum(self._target_q * q_next_online_onehot, 1)
            else:
                q_next_max = tf.reduce_max(self._target_q, 1)
            q_next_max_masked = (1.0 - self._term_ph) * q_next_max
            q_target = self._reward_ph + gamma * q_next_max_masked
            self._td_error = tf.stop_gradient(q_target) - q_selected
            td_error_weighted = self._td_error * self._importance_ph
            self._loss = tf.reduce_mean(tf.square(td_error_weighted), name='loss')
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
        for grad, w in self._grads_vars:
            tf.summary.histogram(w.name, w)
            tf.summary.histogram(w.name + '/gradients', grad)
        if len(self.env.observation_shape) == 1:
            tf.summary.histogram('agent/observation', self._obs)
        elif len(self.env.observation_shape) <= 3:
            tf.summary.image('agent/observation', self._obs)
        else:
            logger.warn('Cannot create summary for observation shape %s' % self.env.obs_shape)
        tf.summary.histogram('agent/action', self._action_onehot)
        tf.summary.histogram('agent/reward_per_action', self._q)
        tf.summary.scalar('agent/learning_rate', self._lr)
        tf.summary.scalar('metrics/loss', self._loss)
        tf.summary.scalar('metrics/train_q', tf.reduce_mean(q_next_max))
        self._summary_op = tf.summary.merge_all()
        self._init_op = tf.global_variables_initializer()

    def _train(self, max_steps, log_dir, render, target_freq, replay, policy, log_freq):
        ep_reward = misc.IncrementalAverage()
        reward_accum = 0
        episode = 0
        writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.sess.run(self._init_op)
        if log_dir and tf.train.latest_checkpoint(log_dir) is not None:
            self.load_weights(log_dir)
        obs = self.env.reset()
        last_time = time.time()
        last_step = self.step_counter
        last_obs = self.obs_counter
        step = self.step_counter
        while step < max_steps:
            self.increment_obs_counter()
            step = self.step_counter
            if render:
                self.env.render()
            reward_per_action = self.predict(obs)
            action = policy.select_action(self.env, reward_per_action, step)
            obs_next, reward, term, info = self.env.step(action)
            reward_accum += reward
            reward = np.clip(reward, -1, 1)
            replay.add(obs, action, reward, obs_next, term)
            obs = obs_next
            if replay.is_ready:
                batch = replay.sample()
                b_obs, b_action, b_reward, b_obs_next, b_term, b_idxs, b_importances = batch
                summarize = term and log_freq and step - last_step > log_freq
                td_error, summary_str = self._train_on_batch(b_obs, b_action, b_reward, b_obs_next,
                                                             b_term, summarize, b_importances)
                try:
                    replay.update(b_idxs, np.abs(td_error))
                except AttributeError:
                    pass
                if step % target_freq == target_freq-1:
                    self.target_update()

                if log_dir and step % log_freq == log_freq-1:
                    self.save_weights(log_dir)

                # Eval & log
                if summarize:
                    step = self.step_counter
                    train_r = ep_reward.reset()
                    test_r, test_q = self.test(episodes=3)
                    obs_per_sec = (self.obs_counter - last_obs) / (time.time() - last_time)
                    step_per_sec = (self.step_counter - last_step) / (time.time() - last_time)
                    last_time = time.time()
                    last_step = step
                    last_obs = self.obs_counter
                    logger.info("On-policy eval.: Average R: %.2f. Step: %d. Ep: %d"
                                % (train_r, step, episode))
                    logger.info("Greedy eval.: Average R: %.2f. "
                                "Average maxQ: %.2f. Step: %d. Ep: %d"
                                % (test_r, test_q, step, episode))
                    logger.info("Performance. Observation/sec: %0.2f. Update/sec: %0.2f."
                                % (obs_per_sec, step_per_sec))
                    if log_dir and summary_str:
                        logs = [tf.Summary.Value(tag='metrics/train_r', simple_value=train_r),
                                tf.Summary.Value(tag='metrics/test_r', simple_value=test_r),
                                tf.Summary.Value(tag='metrics/test_q', simple_value=test_q),
                                tf.Summary.Value(tag='agent/epsilon', simple_value=policy.epsilon),
                                tf.Summary.Value(tag='step/sec', simple_value=step_per_sec),
                                ]
                        writer.add_summary(tf.Summary(value=logs), global_step=step)
                        writer.add_summary(summary_str, global_step=step)
            if term:
                episode += 1
                ep_reward.add(reward_accum)
                reward_accum = 0
                obs = self.env.reset()
        writer.close()

    def _train_on_batch(self, obs, actions, rewards, obs_next,
                        term, summarize=False, importances=None):
        if importances is None:
            importances = [1.0] * len(rewards)
        _, td_error, summary = self.sess.run([self._train_op, self._td_error,
                                              self._summary_op if summarize else self._no_op],
                                             feed_dict={
                                                 self._obs: obs,
                                                 self._action_ph: actions,
                                                 self._reward_ph: rewards,
                                                 self._target_obs: obs_next,
                                                 self._term_ph: term,
                                                 self._importance_ph: importances
                                             })
        return td_error, summary

    def train(self,
              max_steps,
              optimizer,
              learning_rate,
              log_dir,
              replay=ExperienceReplay(capacity=20000, min_size=5000, batch_size=32),
              policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=20000),
              optimizer_args=None,
              decay=None,
              decay_args=None,
              gradient_clip=40.0,
              render=False,
              gamma=0.99,
              target_freq=10000,
              log_freq=10000,
              saver_keep=10,
              **kwargs):
        """Starts training process.

        Args:
            max_steps: (int) Number of training steps (optimizer steps).
            optimizer: An optimizer string name or class.
            learning_rate: (float or Tensor) Optimizer's learning rate.
            log_dir: (str) directory for summary and checkpoints.
                     Continues training, if checkpoint already exists.
            replay: (core.ExperienceReplay) Experience buffer.
            policy: (core.BasePolicy) Agent's training policy.
            optimizer_args: (dict) Keyword arguments, used for optimizer creation.
            decay: (function) Learning rate decay.
                   Expects tensorflow decay function or function name string.
                   Available name strings: 'polynomial', 'exponential'.
                   To disable, pass None.
            decay_args: (dict) Keyword arguments used for learning rate decay function creation.
            gradient_clip: (float) Norm gradient clipping.
                           To disable, pass 0 or None.
            render: (bool) Enables game screen rendering.
            gamma: (float) Reward discount factor.
            target_freq: (int) Target network update frequency (in update steps).
            log_freq: (int) Log and summary frequency (in update steps).
            saver_keep: (int) Maximum number of checkpoints can be stored in `log_dir`.
                        When exceeds, overwrites the most earliest checkpoints.
        """
        self.build_train_graph(optimizer, learning_rate, optimizer_args, gamma,
                               decay, decay_args, gradient_clip, saver_keep)
        try:
            self._train(max_steps, log_dir, render, target_freq, replay, policy, log_freq)
            logger.info('Training finished.')
        except KeyboardInterrupt:
            logger.info('Stopping training process...')
        if log_dir:
            self.save_weights(log_dir)
