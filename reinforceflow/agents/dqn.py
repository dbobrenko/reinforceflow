from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

from six.moves import range  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

import reinforceflow.utils
from reinforceflow.core.base_agent import BaseDQNAgent
from reinforceflow.core import ExperienceReplay
from reinforceflow.core import EGreedyPolicy
from reinforceflow import utils_tf
from reinforceflow import logger
from reinforceflow.utils_tf import add_grads_summary, add_observation_summary


class DQNAgent(BaseDQNAgent):
    def __init__(self, env, net_factory, use_double=True, use_gpu=True, name=''):
        """Constructs Deep Q-Network agent, based on paper:
        "Human-level control through deep reinforcement learning", Mnih et al., 2015.

        See `core.base_agent.BaseDQNAgent.__init__`.
        Args:
            use_double: (bool) Enables Double DQN, described at:
                        "Dueling Network Architectures for Deep Reinforcement Learning",
                        Schaul et al., 2016.
        """
        super(DQNAgent, self).__init__(env=env, net_factory=net_factory, name=name)
        config = tf.ConfigProto(
            device_count={'GPU': use_gpu}
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self._use_double = use_double
        self._importance_ph = tf.placeholder('float32', [None], name='importance_sampling')
        self._td_error = None
        self.opt = None
        self._term_ph = None
        self._target_weights = None
        self._lr = None
        self._action_onehot = None
        self._loss = None
        self._grads = None
        self._grads_vars = None
        self._train_op = None
        self._summary_op = None

    def build_train_graph(self, optimizer, learning_rate, optimizer_args=None, gamma=0.99,
                          decay=None, decay_args=None, gradient_clip=40.0, saver_keep=10):
        """Builds training graph.

        Args:
            optimizer: An optimizer name string or class.
            learning_rate (float or Tensor): Optimizer's learning rate.
            optimizer_args (dict): Keyword arguments used for optimizer creation.
            gamma: (float) Reward discount factor.
            decay: (function) Learning rate decay.
                              Expects tensorflow decay function or function name string.
                              Available name strings: 'polynomial', 'exponential'.
                              To disable, pass None.
            decay_args: (dict) Keyword arguments, passed to the decay function.
            gradient_clip: (float) Norm gradient clipping.
                                   To disable, pass False or None.
            saver_keep: (int) Maximum number of checkpoints can be stored in `log_dir`.
                              When exceeds, overwrites the most earliest checkpoints.
        """
        if self._train_op is not None:
            logger.warn("The training graph has already been built. Skipping.")
            return
        self._term_ph = tf.placeholder('float32', [None], name='term')
        with tf.variable_scope(self._scope + 'target_network') as scope:
            self._target_net =\
                self._net_factory.make(input_shape=[None] + self.env.obs_shape,
                                       output_size=self.env.action_shape[0])
            self._target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope.name)
            self._target_update = [self._target_weights[i].assign(self._weights[i])
                                   for i in range(len(self._target_weights))]

        with tf.variable_scope(self._scope + 'optimizer'):
            self.opt, self._lr = utils_tf.create_optimizer(optimizer, learning_rate,
                                                           optimizer_args=optimizer_args,
                                                           decay=decay, decay_args=decay_args,
                                                           global_step=self.global_step)
            self._action_onehot = tf.arg_max(self._action_ph, 1, name='action_argmax')
            self._action_onehot = tf.one_hot(self._action_onehot, self.env.action_shape[0],
                                             1.0, 0.0, name='action_one_hot')
            # Predict expected future reward for performed action
            q_selected = tf.reduce_sum(self.net.output * self._action_onehot, 1)
            if self._use_double:
                q_next_online_argmax = tf.arg_max(self.net.output, 1)
                q_next_online_onehot = tf.one_hot(q_next_online_argmax,
                                                  self.env.action_shape[0], 1.0)
                q_next_max = tf.reduce_sum(self._target_net.output * q_next_online_onehot, 1)
            else:
                q_next_max = tf.reduce_max(self._target_net.output, 1)
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
        add_grads_summary(self._grads_vars)
        add_observation_summary(self.net.input_ph, self.env.obs_shape)
        tf.summary.histogram('agent/action', self._action_onehot)
        tf.summary.histogram('agent/action_values', self.net.output)
        tf.summary.scalar('metrics/loss', self._loss)
        tf.summary.scalar('agent/learning_rate', self._lr)
        tf.summary.scalar('metrics/avg_q', tf.reduce_mean(q_next_max))
        self._summary_op = tf.summary.merge_all()
        self._init_op = tf.global_variables_initializer()

    def _train(self, max_steps, update_freq, log_dir, render, target_freq, replay,
               policy, log_freq, test_episodes, ignore_checkpoint):
        avg_reward = reinforceflow.utils.IncrementalAverage()
        ep_reward = 0
        episode = 0
        last_log_ep = 0
        writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.sess.run(self._init_op)
        if not ignore_checkpoint and log_dir and tf.train.latest_checkpoint(log_dir) is not None:
            self.load_weights(log_dir)
        obs = self.env.reset()
        last_time = time.time()
        last_step = self.step_counter
        last_obs = self.obs_counter
        step = self.step_counter
        while step < max_steps:
            obs_counter = self.increment_obs_counter()
            step = self.step_counter
            if render:
                self.env.render()
            action_values = self.predict_on_batch([obs])
            action = policy.select_action(self.env, action_values, step)
            obs_next, reward, term, info = self.env.step(action)
            ep_reward += reward
            reward = np.clip(reward, -1, 1)
            replay.add(obs, action, reward, obs_next, term)
            obs = obs_next
            if replay.is_ready and obs_counter % update_freq == 0:
                batch = replay.sample()
                b_obs, b_action, b_reward, b_obs_next, b_term, b_idxs, b_importances = batch
                summarize = episode > last_log_ep and step - last_step > log_freq
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
                    last_log_ep = episode
                    step = self.step_counter
                    num_ep = avg_reward.length
                    max_r = avg_reward.max
                    min_r = avg_reward.min
                    train_r = avg_reward.reset()
                    test_r = self.test(episodes=test_episodes, copy_env=True).compute_average()
                    obs_per_sec = (self.obs_counter - last_obs) / (time.time() - last_time)
                    step_per_sec = (self.step_counter - last_step) / (time.time() - last_time)
                    last_time = time.time()
                    last_step = step
                    last_obs = self.obs_counter
                    logger.info("On-policy eval.: Average R: %.2f. Step: %d. Ep: %d"
                                % (train_r, step, episode))
                    logger.info("Greedy eval.: Average R: %.2f. Step: %d. Ep: %d"
                                % (test_r, step, episode))
                    logger.info("Performance. Observation/sec: %0.2f. Update/sec: %0.2f."
                                % (obs_per_sec, step_per_sec))
                    if log_dir and summary_str:
                        logs = [tf.Summary.Value(tag='metrics/total_ep', simple_value=episode),
                                tf.Summary.Value(tag='metrics/num_ep', simple_value=num_ep),
                                tf.Summary.Value(tag='metrics/max_r', simple_value=max_r),
                                tf.Summary.Value(tag='metrics/min_r', simple_value=min_r),
                                tf.Summary.Value(tag='metrics/avg_r', simple_value=train_r),
                                tf.Summary.Value(tag='metrics/test_r', simple_value=test_r),
                                tf.Summary.Value(tag='agent/epsilon', simple_value=policy.epsilon),
                                tf.Summary.Value(tag='step/sec', simple_value=step_per_sec),
                                ]
                        writer.add_summary(tf.Summary(value=logs), global_step=step)
                        writer.add_summary(summary_str, global_step=step)
            if term:
                episode += 1
                avg_reward.add(ep_reward)
                ep_reward = 0
                obs = self.env.reset()
        writer.close()

    def _train_on_batch(self, obs, actions, rewards, obs_next,
                        term, summarize=False, importance=None):
        if importance is None:
            importance = [1.0] * len(rewards)
        _, td_error, summary = self.sess.run([self._train_op, self._td_error,
                                              self._summary_op if summarize else self._no_op],
                                             feed_dict={
                                                 self.net.input_ph: obs,
                                                 self._action_ph: actions,
                                                 self._reward_ph: rewards,
                                                 self._target_net.input_ph: obs_next,
                                                 self._term_ph: term,
                                                 self._importance_ph: importance
                                             })
        return td_error, summary

    def train(self,
              max_steps,
              optimizer,
              learning_rate,
              log_dir,
              update_freq=1,
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
              saver_keep=3,
              test_episodes=3,
              ignore_checkpoint=False,
              **kwargs):
        """Starts training process.

        Args:
            max_steps: (int) Number of training steps (optimizer steps).
            update_freq: (int) Optimizer update frequency.
            optimizer: An optimizer string name or class.
            learning_rate: (float or Tensor) Optimizer learning rate.
            log_dir: (str) Directory used for summary and checkpoints.
                     Continues training, if checkpoint already exists.
            replay: (core.ExperienceReplay) Experience buffer.
            policy: (core.BasePolicy) Agent's training policy.
            optimizer_args: (dict) Keyword arguments used for optimizer creation.
            decay: (function) Learning rate decay.
                   Expects tensorflow decay function or function name string.
                   Available names: 'polynomial', 'exponential'.
                   To disable, pass None.
            decay_args: (dict) Keyword arguments used for learning rate decay function creation.
            gradient_clip: (float) Norm gradient clipping. To disable, pass 0 or None.
            render: (bool) Enables game screen rendering.
            gamma: (float) Reward discount factor.
            target_freq: (int) Target network update frequency (in update steps).
            log_freq: (int) Checkpoint and summary saving frequency (in update steps).
            saver_keep: (int) Maximum number of checkpoints can be stored in `log_dir`.
                        When exceeds, overwrites the most earliest checkpoints.
            test_episodes: (int) Number of test episodes.
            ignore_checkpoint: (bool) If enabled, training will start from scratch,
                               and overwrite all old checkpoints found at `log_dir` path.
        """
        self.build_train_graph(optimizer, learning_rate, optimizer_args, gamma,
                               decay, decay_args, gradient_clip, saver_keep)
        try:
            self._train(max_steps, update_freq, log_dir, render, target_freq, replay,
                        policy, log_freq, test_episodes, ignore_checkpoint)
            logger.info('Training finished.')
        except KeyboardInterrupt:
            logger.info('Stopping training process...')
        if log_dir:
            self.save_weights(log_dir)
