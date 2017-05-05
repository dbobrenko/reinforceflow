from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import time
from six.moves import range
import os
import numpy as np
import tensorflow as tf
from reinforceflow.agents import DiscreteAgent
from reinforceflow.core import ExperienceReplay
from reinforceflow.nets import dqn
from reinforceflow.core import EGreedyPolicy, GreedyPolicy
from reinforceflow import misc
from reinforceflow import logger
# TODO: Test & write unittests
# TODO: Add comments & documentation
# TODO: Log more info to tensorboard
# TODO: Make Environment Factory
# TODO: Make preprocessing function (in graph)
# TODO: Remove train_on_batch from public methods or add setup/compile method
# TODO: Make Base TFAgent or DeepAgent class


class DQNAgent(DiscreteAgent):
    def __init__(self,
                 env,
                 sess=None,
                 net_fn=dqn,
                 gradient_clip=40.0,
                 opt=tf.train.RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01),
                 learning_rate=None,
                 decay=None,
                 decay_poly_end_lr=0.0001,
                 decay_poly_power=1.0,
                 decay_poly_steps=1e7,
                 decay_rate=0.96):
        """Constructs Deep Q-Network agent;
        Based on paper "Human-level control through deep reinforcement learning", Mnih et al., 2015.
        Current agent solves environments with discrete _action spaces. Initially designed to work with raw pixel inputs.

        Args:
            env (reinforceflow.EnvWrapper): Environment wrapper
            sess: TensorFlow Session
            net_fn: Function, that takes `input_shape` and `output_size` arguments,
                    and returns (input Tensor, output Tensor, all end point Tensors)
            epsilon (float): The probability for epsilon-greedy exploration, expected to be in range [0; 1]
            gradient_clip (float): Norm gradient clipping, to disable, pass 0 or None
            log_dir (str): path to directory, where all agent's outputs will be saved (session, summary, logs, etc)
            opt: An optimizer instance, optimizer name, or optimizer class
            learning_rate (float or Tensor): Should be provided, if `opt` is optimizer class or name
            decay: Learning rate decay. Should be provided decay function, or decay function name.
                   Available decays: 'polynomial', 'exponential'. To disable decay, pass None.
            decay_poly_end_lr (float or Tensor): The minimal end learning rate.
                                                 Should be provided, if polynomial decay was chosen.
            decay_poly_power (float or Tensor): The power of the polynomial.
                                                E.g. `power=1.0` means linear learning rate decay.
                                                Should be provided, if polynomial decay was chosen.
            decay_poly_steps (int or Tensor): The number of steps over which learning rate anneals
                                              down to `decay_poly_end_lr`.
                                              Should be provided, if polynomial decay was chosen.
            decay_rate (float): The decay rate. Should be provided, if exponential decay was chosen.

        Attributes:
            env: Agent's running environment
            sess: TensorFlow Session
            opt: TensorFlow optimizer
            lr: TensorFlow optimizer's learning rate
        """
        super(DQNAgent, self).__init__(env=env)
        self.sess = tf.Session() if sess is None else sess
        with tf.variable_scope('network'):
            self._action = tf.placeholder('int32', [None], name='action')
            self._reward = tf.placeholder('float32', [None], name='reward')
            self._obs, self._q, _ = net_fn(input_shape=[None] + self.env.obs_shape, output_size=self.env.action_shape)

        with tf.variable_scope('target_network'):
            self._target_obs, self._target_q, _ = net_fn(input_shape=[None] + self.env.obs_shape,
                                                         output_size=self.env.action_shape)

        with tf.variable_scope('target_update'):
            target_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_network')
            w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'network')
            self._target_update = [target_w[i].assign(w[i]) for i in range(len(target_w))]

        with tf.variable_scope('optimizer'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.opt, self.lr = misc.create_optimizer(opt, learning_rate, decay=decay,
                                                      global_step=self.global_step,
                                                      decay_poly_steps=decay_poly_steps,
                                                      decay_poly_end_lr=decay_poly_end_lr,
                                                      decay_poly_power=decay_poly_power,
                                                      decay_rate=decay_rate)
            action_one_hot = tf.one_hot(self._action, self.env.action_shape, 1.0, 0.0, name='action_one_hot')
            # Predict expected future reward for performed action
            q_value = tf.reduce_sum(tf.multiply(self._q, action_one_hot), axis=1)
            self._loss = tf.reduce_mean(tf.square(self._reward - q_value), name='loss')
            grads = tf.gradients(self._loss, w)
            if gradient_clip:
                grads, _ = tf.clip_by_global_norm(grads, gradient_clip)
            grads_vars = list(zip(grads, w))
            self._train_op = self.opt.apply_gradients(grads_vars, global_step=self.global_step)
        self._saver = tf.train.Saver(max_to_keep=30)

        # Summaries
        self._writer = None
        self._no_op = tf.no_op()
        for grad, w in grads_vars:
            tf.summary.histogram(w.name, w)
            tf.summary.histogram('gradients/' + w.name, grad)
        if len(self.env.obs_shape) == 1:
            tf.summary.histogram('agent/observation', self._obs)
        elif len(self.env.obs_shape) <= 3:
            tf.summary.image('agent/observation', self._obs)
        else:
            logger.warn('Cannot create summary for observation with shape %s' % self.env.obs_shape)
        tf.summary.histogram('agent/action', action_one_hot)
        tf.summary.histogram('agent/reward_per_action', self._q)
        tf.summary.scalar('agent/learning_rate', self.lr)
        tf.summary.scalar('metrics/loss', self._loss)
        self._summary_op = tf.summary.merge_all()
        # TODO print final config

    def predict(self, obs):
        return self.sess.run(self._q, {self._obs: obs})

    def predict_target(self, obs):
        return self.sess.run(self._target_q, {self._target_obs: obs})

    def update_target(self):
        self.sess.run(self._target_update)

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

    def train_on_batch(self, obs, actions, rewards, summarize=False):
        """Trains agent on given transitions batch.

        Args:
            obs (nd.array): input observations with shape=[batch, height, width, channels]
            actions: list of actions
            rewards: list with rewards for each action
            summarize: if enabled, writes summaries into TensorBoard
        """
        _, summary = self.sess.run([self._train_op, self._summary_op if summarize else self._no_op], feed_dict={
                                    self._obs: obs,
                                    self._action: actions,
                                    self._reward: rewards
                                    })
        return summary

    def test(self, episodes, policy=GreedyPolicy(), max_ep_steps=1e5, render=False):
        """Tests agent's performance with specified policy on given number of games"""
        ep_rewards = misc.IncrementalAverage()
        ep_q = misc.IncrementalAverage()
        for _ in range(episodes):
            reward_accum = 0
            obs = self.env.reset()
            for _ in range(int(max_ep_steps)):
                if render:
                    self.env.render()
                reward_per_action = self.predict(obs)
                action = policy.select_action(reward_per_action, env=self.env)
                obs, r, terminal, info = self.env.step(action)
                ep_q.add(np.max(reward_per_action))
                reward_accum += r
                if terminal:
                    break
            ep_rewards.add(reward_accum)
        return ep_rewards.compute_average(), ep_q.compute_average()

    def train(self,
              max_steps,
              log_dir,
              render=False,
              target_freq=10000,
              gamma=0.99,
              checkpoint=None,
              experience=ExperienceReplay(size=1000000, batch_size=32, min_size=50000),
              policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=1000000),
              log_freq=100,
              checkpoint_freq=20000):
        ep_reward = misc.IncrementalAverage()
        ep_q = misc.IncrementalAverage()
        reward_accum = 0
        last_log_step = 0
        episode = 0
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            self._writer = tf.summary.FileWriter(log_dir, self.sess.graph)
            if checkpoint:
                self.load_weights(checkpoint)
            obs = self.env.reset()
            try:
                last_time = time.time()
                last_step = self.sess.run(self.global_step)
                for _ in range(int(max_steps)):
                    step = self.sess.run(self.global_step)
                    if render:
                        self.env.render()
                    reward_per_action = self.predict(obs)
                    action = policy.select_action(reward_per_action, self.env, step)
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
                logger.info('Training finished.')
            except KeyboardInterrupt:
                logger.info('Stopping training process...')
                if log_dir:
                    self.save_weights(log_dir)
                sys.exit(0)
            if log_dir:
                self.save_weights(log_dir)
