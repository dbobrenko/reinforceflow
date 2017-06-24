from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import time
import os

from six.moves import range  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

from reinforceflow.core.base_agent import BaseDQNAgent
from reinforceflow.core import ExperienceReplay
from reinforceflow.nets import dqn
from reinforceflow.core import EGreedyPolicy
from reinforceflow import misc
from reinforceflow import logger
# TODO: Test & write unittests
# TODO: Add comments & documentation


class DQNAgent(BaseDQNAgent):
    def __init__(self,
                 env,
                 optimizer,
                 learning_rate,
                 net_fn=dqn,
                 optimizer_args=None,
                 decay=None,
                 decay_args=None,
                 gradient_clip=40.0,
                 name=''):
        """Constructs Deep Q-Network agent, based on paper
        "Human-level control through deep reinforcement learning", Mnih et al., 2015.
        Current agent solves environments with discrete action spaces. Initially designed to work with raw pixel inputs.

        Args:
            env (reinforceflow.EnvWrapper): Environment wrapper.
            optimizer: An optimizer string name or class.
            learning_rate (float or Tensor): Should be provided, if `opt` is optimizer class or name.
            net_fn: Function, that takes `input_shape` and `output_size` arguments,
                    and returns tuple(input Tensor, output Tensor, all end point Tensors).
            optimizer_args (dict): kwargs used for optimizer creation.
                                   If None, RMSProp args are applied: momentum=0.95, epsilon=0.01
            decay: Learning rate decay. Should be provided decay function, or decay function name.
                   Available decays: 'polynomial', 'exponential'. To disable decay, pass None.
            decay_args (dict): kwargs used for learning rate decay function creation.
            gradient_clip (float): Norm gradient clipping, to disable, pass 0 or None.

        Attributes:
            env: Agent's running environment
            optimizer: TensorFlow optimizer
        """
        super(DQNAgent, self).__init__(env=env, net_fn=net_fn, optimizer=optimizer, learning_rate=learning_rate,
                                       optimizer_args=optimizer_args, gradient_clip=gradient_clip, decay=decay,
                                       decay_args=decay_args, name=name)
        # Summaries
        self._writer = None
        for grad, w in self._grads_vars:
            tf.summary.histogram(w.name, w)
            tf.summary.histogram(w.name + '/gradients', grad)
        if len(self.env.observation_shape) == 1:
            tf.summary.histogram('agent/observation', self._obs)
        elif len(self.env.observation_shape) <= 3:
            tf.summary.image('agent/observation', self._obs)
        else:
            logger.warn('Cannot create summary for observation with shape %s' % self.env.obs_shape)
        tf.summary.histogram('agent/action', self._action_one_hot)
        tf.summary.histogram('agent/reward_per_action', self._q)
        tf.summary.scalar('agent/learning_rate', self.lr)
        tf.summary.scalar('metrics/loss', self._loss)
        self._summary_op = tf.summary.merge_all()
        # TODO print final config

    def save_weights(self, path, model_name='model.ckpt'):
        if not os.path.exists(path):
            os.makedirs(path)
        self._saver.save(self.sess, os.path.join(path, model_name), global_step=self.global_step)
        logger.info('Checkpoint saved to %s' % os.path.join(path, model_name))

    def load_weights(self, checkpoint):
        if not os.path.exists(checkpoint):
            raise ValueError('Checkpoint path/dir %s does not exists.' % checkpoint)
        if tf.gfile.IsDirectory(checkpoint):
            checkpoint = tf.train.latest_checkpoint(checkpoint)
        logger.info('Restoring checkpoint from %s', checkpoint)
        self._saver.restore(self.sess, save_path=checkpoint)
        self.update_target()

    def _train(self,
               max_steps,
               log_dir,
               render,
               target_freq,
               gamma,
               checkpoint,
               experience,
               policy,
               log_freq,
               checkpoint_freq):
        ep_reward = misc.IncrementalAverage()
        ep_q = misc.IncrementalAverage()
        reward_accum = 0
        last_log_step = 0
        episode = 0
        self.sess.run(tf.global_variables_initializer())
        self._writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        if checkpoint:
            self.load_weights(checkpoint)
        obs = self.env.reset()
        last_time = time.time()
        last_step = self.current_step
        for _ in range(max_steps):
            step = self.current_step
            if render:
                self.env.render()
            reward_per_action = self.predict(obs)
            action = policy.select_action(self.env, reward_per_action, step)
            obs_next, reward, term, info = self.env.step(action)
            reward_accum += reward
            reward = np.clip(reward, -1, 1)
            experience.add({'obs': obs, 'action': action, 'reward': reward, 'obs_next': obs_next, 'term': term})
            obs = obs_next

            # Update step:
            if experience.is_ready:
                batch = experience.sample()
                tr_obs = []
                tr_action = []
                tr_reward = []
                for transition in batch:
                    tr_obs.append(transition['obs'])
                    tr_action.append(transition['action'])
                    td_target = transition['reward']
                    if not transition['term']:
                        q = np.max(self.predict_target(transition['obs_next']).flatten())
                        td_target += gamma * q
                        ep_q.add(q)
                    tr_reward.append(td_target)
                summarize = term and log_freq and step - last_log_step > log_freq
                summary_str = self.train_on_batch(np.vstack(tr_obs), tr_action, tr_reward, summarize)

                if step % target_freq == target_freq-1:
                    self.update_target()

                if log_dir and step % checkpoint_freq == checkpoint_freq-1:
                    self.save_weights(log_dir)

                # Eval & log
                if summarize:
                    last_log_step = step
                    train_r = ep_reward.reset()
                    train_q = ep_q.reset()
                    test_r, test_q = self.test(episodes=3)
                    logger.info("Train. Average Ep Reward: %.2f. Average Q value: %.2f. Step: %d. Ep: %d"
                                % (train_r, train_q, step, episode))
                    logger.info("Test. Average Ep Reward: %.2f. Average Q value: %.2f. Step: %d. Ep: %d"
                                % (test_r, test_q, step, episode))
                    if log_dir and summary_str:
                        step_per_sec = (step - last_step) / (time.time() - last_time)
                        last_time = time.time()
                        last_step = step
                        custom_values = [tf.Summary.Value(tag='metrics/train_r', simple_value=train_r),
                                         tf.Summary.Value(tag='metrics/train_q', simple_value=train_q),
                                         tf.Summary.Value(tag='metrics/test_r', simple_value=test_r),
                                         tf.Summary.Value(tag='metrics/test_q', simple_value=test_q),
                                         tf.Summary.Value(tag='agent/epsilon', simple_value=policy.epsilon),
                                         tf.Summary.Value(tag='step/sec', simple_value=step_per_sec),
                                         ]
                        self._writer.add_summary(tf.Summary(value=custom_values), global_step=step)
                        self._writer.add_summary(summary_str, global_step=step)
            if term:
                episode += 1
                ep_reward.add(reward_accum)
                reward_accum = 0
                obs = self.env.reset()

    def train(self,
              log_dir,
              max_steps,
              render=False,
              target_freq=10000,
              gamma=0.99,
              checkpoint=None,
              experience=ExperienceReplay(size=1000000, min_size=50000, batch_size=32),
              policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=1000000),
              log_freq=100,
              checkpoint_freq=20000):
        try:
            self._train(max_steps, log_dir, render, target_freq, gamma, checkpoint, experience, policy, log_freq,
                        checkpoint_freq)
            logger.info('Training finished.')
        except KeyboardInterrupt:
            logger.info('Stopping training process...')
            if log_dir:
                self.save_weights(log_dir)
            sys.exit(0)
        if log_dir:
            self.save_weights(log_dir)
