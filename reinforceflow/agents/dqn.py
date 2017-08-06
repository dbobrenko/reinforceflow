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

    def _build_train_graph(self, optimizer, learning_rate, optimizer_args=None,
                           decay=None, decay_args=None, gradient_clip=40.0, saver_keep=10):
        super(DQNAgent, self)._build_train_graph(optimizer, learning_rate,
                                                 optimizer_args, decay, decay_args,
                                                 gradient_clip, saver_keep)
        for grad, w in self._grads_vars:
            tf.summary.histogram(w.name, w)
            tf.summary.histogram(w.name + '/gradients', grad)
        if len(self.env.observation_shape) == 1:
            tf.summary.histogram('agent/observation', self._obs)
        elif len(self.env.observation_shape) <= 3:
            tf.summary.image('agent/observation', self._obs)
        else:
            logger.warn('Cannot create summary for observation shape %s' % self.env.obs_shape)
        tf.summary.histogram('agent/action', self._action_one_hot)
        tf.summary.histogram('agent/reward_per_action', self._q)
        tf.summary.scalar('agent/learning_rate', self._lr)
        tf.summary.scalar('metrics/loss', self._loss)
        self._summary_op = tf.summary.merge_all()
        self._init_op = tf.global_variables_initializer()

    def _train(self, max_steps, log_dir, render, target_freq, gamma, replay, policy, log_freq):
        ep_reward = misc.IncrementalAverage()
        ep_q = misc.IncrementalAverage()
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
                batch_obs, batch_action, batch_reward, batch_o_next, batch_term, batch_idxs = batch
                q_values = self.target_predict(batch_o_next)
                if self._use_double:
                    idx_mask = np.zeros_like(q_values, dtype=np.bool)
                    action_idxs = np.argmax(self.predict(batch_o_next), axis=1)
                    for i, idx in enumerate(action_idxs):
                        idx_mask[i, idx] = True
                    q_values = q_values[idx_mask]
                else:
                    q_values = np.max(q_values, axis=1)
                ep_q.add_batch(q_values)
                batch_reward += (1 - batch_term) * gamma * q_values
                summarize = term and log_freq and step - last_step > log_freq
                td_error, summary_str = self._train_on_batch(batch_obs, batch_action,
                                                             batch_reward, summarize)
                try:
                    replay.update(td_error, batch_idxs)
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
                    train_q = ep_q.reset()
                    test_r, test_q = self.test(episodes=3)
                    obs_per_sec = (self.obs_counter - last_obs) / (time.time() - last_time)
                    step_per_sec = (self.step_counter - last_step) / (time.time() - last_time)
                    last_time = time.time()
                    last_step = step
                    last_obs = self.obs_counter
                    logger.info("On-policy eval.: Average R: %.2f. "
                                "Average maxQ: %.2f. Step: %d. Ep: %d"
                                % (train_r, train_q, step, episode))
                    logger.info("Greedy eval.: Average R: %.2f. "
                                "Average maxQ: %.2f. Step: %d. Ep: %d"
                                % (test_r, test_q, step, episode))
                    logger.info("Performance. Observation/sec: %0.2f. Update/sec: %0.2f."
                                % (obs_per_sec, step_per_sec))
                    if log_dir and summary_str:
                        logs = [tf.Summary.Value(tag='metrics/train_r', simple_value=train_r),
                                tf.Summary.Value(tag='metrics/train_q', simple_value=train_q),
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

    def train(self,
              max_steps,
              optimizer,
              learning_rate,
              log_dir,
              experience=ExperienceReplay(capacity=20000, min_size=5000, batch_size=32),
              policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=20000),
              optimizer_args=None,
              decay=None,
              decay_args=None,
              gradient_clip=40.0,
              render=False,
              gamma=0.99,
              target_freq=10000,
              log_freq=2000,
              saver_keep=10):
        """Starts training process.

        Args:
            max_steps: (int) Number of training steps (optimizer steps).
            optimizer: An optimizer string name or class.
            learning_rate: (float or Tensor) Optimizer's learning rate.
            log_dir: (str) directory for summary and checkpoints.
                     Continues training, if checkpoint already exists.
            experience: (core.ExperienceReplay) Experience buffer.
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
        self._build_train_graph(optimizer, learning_rate, optimizer_args,
                                decay, decay_args, gradient_clip, saver_keep)
        try:
            self._train(max_steps, log_dir, render, target_freq, gamma,
                        experience, policy, log_freq)
            logger.info('Training finished.')
        except KeyboardInterrupt:
            logger.info('Stopping training process...')
        if log_dir:
            self.save_weights(log_dir)
