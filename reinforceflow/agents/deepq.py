from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from reinforceflow import logger
from reinforceflow.core import ProportionalReplay, EGreedyPolicy, Stats, losses
from reinforceflow.core.agent import BaseDeepQ
from reinforceflow.core.optimizer import Optimizer, RMSProp
from reinforceflow.core.schedule import Schedule
from reinforceflow.utils import tensor_utils


class DeepQ(BaseDeepQ):
    def __init__(self, env, model, use_double=True, restore_from=None, device='/gpu:0',
                 optimizer=None, saver_keep=3, name='DeepQ'):
        """Constructs Deep Q-Learning agent.
         Includes the following implementations:
            1. Human-level control through deep reinforcement learning, Mnih et al., 2015.
            2. Dueling Network Architectures for Deep Reinforcement Learning, Wang et al., 2015.
                See `models.DeepQModel`.
            3. Deep Reinforcement Learning with Double Q-learning, Hasselt et al., 2016.
                See `use_double` argument.
            4. Prioritized Experience Replay, Schaul et al., 2015.
                See `core.replay.ProportionalReplay`.

        See `core.BaseDeepQ`.
        Args:
            env (gym.Env): Environment instance.
            model (models.AbstractFactory): Network factory.
            use_double (bool): Enables Double DQN.
            restore_from (str): Path to the pre-trained model.
            device (str): TensorFlow device, used for graph creation.
            optimizer (str or Optimizer): [Training-only] Agent's optimizer.
                By default: RMSProp(lr=2.5e-4, momentum=0.95).
            saver_keep (int): [Training-only] Maximum number of checkpoints can be stored at once.
        """
        super(DeepQ, self).__init__(env=env, model=model, device=device,
                                    saver_keep=saver_keep, name=name)
        self._use_double = use_double
        self._last_log_time = time.time()
        self._last_target_sync = self.step

        if optimizer is None:
            optimizer = RMSProp(0.00025, momentum=0.95, epsilon=0.01)

        with tf.device(self.device):
            self._importance_ph = tf.placeholder('float32', [None], name='importance_sampling')
            self._term_ph = tf.placeholder('float32', [None], name='term')
            self._gamma_ph = tf.placeholder('float32', [], name='gamma')
            with tf.variable_scope(self._scope + 'optimizer'):
                if self._use_double:
                    q_idx = tf.argmax(self.net['out'], 1)
                    q_onehot = tf.one_hot(q_idx, self.env.action_space.shape[0], 1.0)
                    q_next_max = tf.reduce_sum(self._target_net['out'] * q_onehot, 1)
                else:
                    q_next_max = tf.reduce_max(self._target_net['out'], 1)

                # Loss:
                q_next_masked = (1.0 - self._term_ph) * q_next_max
                target = self._reward_ph + self._gamma_ph * q_next_masked
                loss, self._td = losses.td_error_q(q_logits=self.net['out'],
                                                   action=self._action_ph,
                                                   target=tf.stop_gradient(target),
                                                   weights=self._importance_ph,
                                                   name='loss')
                self.opt = Optimizer.create(optimizer)
                self.opt.build(self.global_step)
                self._train_op = self.opt.minimize(loss, self._weights)
        tensor_utils.add_observation_summary(self.net['in'], self.env)
        tf.summary.histogram('agent/action', self._action_ph)
        tf.summary.histogram('agent/action_values', self.net['out'])
        tf.summary.scalar('agent/learning_rate', self.opt.lr_ph)
        tf.summary.scalar('metrics/loss', loss)
        tf.summary.scalar('metrics/avg_Q', tf.reduce_mean(q_next_max))
        self._summary_op = tf.summary.merge_all()
        self._savings |= set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               self._scope + 'optimizer'))
        self._saver = tf.train.Saver(var_list=list(self._savings), max_to_keep=saver_keep)
        self.sess.run(tf.global_variables_initializer())
        if restore_from and tf.train.latest_checkpoint(restore_from):
            self.load_weights(restore_from)

    def train_on_batch(self, obs, actions, rewards, obs_next,
                       term, lr, gamma=0.99, summarize=False, importance=None):
        if importance is None:
            importance = np.ones_like(rewards)
        _, td_error, summary = self.sess.run([self._train_op, self._td,
                                              self._summary_op if summarize else self._no_op],
                                             feed_dict={
                                                 self.net['in']: obs,
                                                 self._action_ph: actions,
                                                 self._reward_ph: rewards,
                                                 self._target_net['in']: obs_next,
                                                 self._term_ph: term,
                                                 self._importance_ph: importance,
                                                 self.opt.lr_ph: lr,
                                                 self._gamma_ph: gamma
                                             })
        return td_error, summary

    def train(self, maxsteps, log_dir, log_freq, log_on_term=True, lr_schedule=None, gamma=0.99,
              policy=None, target_freq=40000, update_freq=4, replay=None, render=False,
              test_env=None, test_render=False, test_episodes=1, test_maxsteps=1000):
        """Starts training.

        Args:
            maxsteps (int): Total amount of seen observations.
            log_dir (str): Path used for summary and checkpoints.
            log_freq (int): Checkpoint and summary saving frequency (in seconds).
            log_on_term (bool): Whether to log only after episode ends.
            lr_schedule (core.Schedule): Learning rate scheduler.
            gamma (float): Reward discount factor.
            policy (core.BasePolicy): Agent's training policy.
            target_freq (int): Target network update frequency(in seen observations).
            update_freq (int) Optimizer update frequency (in seen observations).
            replay (core.ExperienceReplay): Experience replay buffer.
            render (bool): Enables game screen rendering.
            test_env (gym.Env): Environment instance, used for testing.
            test_render (bool): Enables rendering for test evaluations.
            test_episodes (int): Number of test episodes. To disable test evaluation, pass 0.
            test_maxsteps (int): Maximum step allowed during test per episode.
        """
        try:
            lr_schedule = Schedule.create(lr_schedule, self.opt.learning_rate, maxsteps)
            replay = replay if replay else ProportionalReplay(50000, 32, 10000)
            policy = policy if policy else EGreedyPolicy(1.0, 0.1, maxsteps / 50)

            writer = tf.summary.FileWriter(log_dir, self.sess.graph)
            stats = Stats(log_freq=log_freq, log_on_term=log_on_term, file_writer=writer,
                          log_prefix='Train')
            obs = self.env.reset()
            # Play & Data Collection loop
            while self.step < maxsteps:
                if render:
                    self.env.render()
                action_values = self.predict_on_batch([obs])
                action = policy.select_action(self.env, action_values, self.step)

                obs_next, reward, term, info = self.env.step(action)
                self.step += 1
                self.episode += int(term)
                stats.add(reward, term, info, step=self.step, episode=self.episode)
                replay.add(obs, action, reward, obs_next, term)
                obs = obs_next
                if term:
                    obs = self.env.reset()

                # Train from experience replay, if ready
                self.train_from_replay(log_dir=log_dir,
                                       log_freq=log_freq,
                                       lr=lr_schedule.value(self.step),
                                       gamma=gamma,
                                       target_freq=target_freq,
                                       update_freq=update_freq,
                                       replay=replay,
                                       test_env=test_env,
                                       test_episodes=test_episodes,
                                       test_render=test_render,
                                       test_maxsteps=test_maxsteps,
                                       writer=writer)
            logger.info('Performing final evaluation.')
            self.test(test_env, test_episodes, max_steps=test_maxsteps, render=test_render)
            writer.close()
            logger.info('Training finished.')
        except KeyboardInterrupt:
            logger.info('Stopping training process...')
        self.save_weights(log_dir)

    def train_from_replay(self, log_dir, log_freq, lr, gamma, target_freq, update_freq, replay,
                          test_env, test_episodes, test_render, test_maxsteps, writer):
        if not replay.is_ready:
            return

        if self.step % update_freq != 0:
            return 

        b_obs, b_action, b_reward, b_obs_next, b_term, b_idxs, b_is = replay.sample()
        summarize = time.time() - self._last_log_time > log_freq
        td_error, summary_str = self.train_on_batch(b_obs, b_action, b_reward, b_obs_next,
                                                    b_term,
                                                    lr=lr,
                                                    gamma=gamma,
                                                    summarize=summarize,
                                                    importance=b_is)

        if isinstance(replay, ProportionalReplay):
            replay.update(b_idxs, np.abs(td_error))

        if self.step - self._last_target_sync > target_freq:
            self._last_target_sync = self.step
            self.target_update()

        if summarize:
            self.save_weights(log_dir)
            self._last_log_time = time.time()
            self.test(test_env, test_episodes, max_steps=test_maxsteps, render=test_render)
            if log_dir and summary_str:
                writer.add_summary(summary_str, global_step=self.step)
                writer.flush()

