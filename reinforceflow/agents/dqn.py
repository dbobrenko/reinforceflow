from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import reinforceflow.utils
from reinforceflow import logger
from reinforceflow.core import ProportionalReplay, EGreedyPolicy
from reinforceflow.core.agent import BaseDQNAgent
from reinforceflow.core.optimizer import Optimizer, RMSProp
from reinforceflow.utils import tensor_utils


class DQNAgent(BaseDQNAgent):
    def __init__(self, env, net_factory, use_double=True, restore_from=None, device='/gpu:0',
                 steps=int(32*50e6), optimizer=None, policy=None, gamma=0.99, exp=None,
                 target_freq=32*10000, update_freq=4, saver_keep=3, log_every_sec=600, name='DQN'):
        """Constructs Deep Q-Network agent.
         Includes the following implementations:
            1. Human-level control through deep reinforcement learning, Mnih et al., 2015.
            2. Dueling Network Architectures for Deep Reinforcement Learning, Wang et al., 2015.
                See `nets.dueling`.
            3. Deep Reinforcement Learning with Double Q-learning, Hasselt et al., 2016.
                See `use_double` argument.
            4. Prioritized Experience Replay, Schaul et al., 2015.
                See `core.replay.ProportionalReplay`.

        See `core.BaseDQNAgent`.
        Args:
            env (gym.Env): Environment instance.
            net_factory (nets.AbstractFactory): Network factory.
            use_double (bool): Enables Double DQN.
            restore_from (str): Path to the pre-trained model.
            device (str): TensorFlow device, used for graph creation.
            steps (int): [Training-only] Total amount of seen observations.
            optimizer (str or Optimizer): [Training-only] Agent's optimizer.
                By default: RMSProp(lr=2.5e-4, momentum=0.95).
            policy (core.BasePolicy): [Training-only] Agent's training policy.
            gamma (float): [Training-only] Reward discount factor.
            exp: (core.ExperienceReplay) [Training-only] Experience replay buffer.
            target_freq (int): [Training-only] Target network update frequency
                (in seen observations).
            update_freq (int): [Training-only] Optimizer update frequency
                (in seen observations).
            saver_keep (int): [Training-only] Maximum number of checkpoints can be stored at once.
            log_every_sec (int): [Training-only] Checkpoint and summary saving frequency
                (in seconds).
        """
        super(DQNAgent, self).__init__(env=env, net_factory=net_factory, device=device,
                                       saver_keep=saver_keep, name=name)
        self._steps = steps
        self._policy = policy
        self._exp = exp
        self._target_freq = target_freq
        self._update_freq = update_freq
        self._log_freq = log_every_sec
        self._use_double = use_double

        if optimizer is None:
            optimizer = RMSProp(0.00025, momentum=0.95, epsilon=0.01, lr_decay='linear')

        if self._exp is None:
            self._exp = ProportionalReplay(50000, 32, 32)

        if self._policy is None:
            self._policy = EGreedyPolicy(1.0, 0.1, steps / 50)

        with tf.device(self.device):
            self._importance_ph = tf.placeholder('float32', [None], name='importance_sampling')
            self._term_ph = tf.placeholder('float32', [None], name='term')
            with tf.variable_scope(self._scope + 'optimizer'):
                action_onehot = tf.argmax(self._action_ph, 1, name='action_argmax')
                action_onehot = tf.one_hot(action_onehot, self.env.action_space.shape[0],
                                           1.0, 0.0, name='action_one_hot')
                # Predict expected future reward for performed action
                q_selected = tf.reduce_sum(self.net.output * action_onehot, 1)
                if self._use_double:
                    q_next_online_argmax = tf.argmax(self.net.output, 1)
                    q_next_online_onehot = tf.one_hot(q_next_online_argmax,
                                                      self.env.action_space.shape[0], 1.0)
                    q_next_max = tf.reduce_sum(self._target_net.output * q_next_online_onehot, 1)
                else:
                    q_next_max = tf.reduce_max(self._target_net.output, 1)
                q_next_max_masked = (1.0 - self._term_ph) * q_next_max
                q_target = self._reward_ph + gamma * q_next_max_masked
                self._td_error = tf.stop_gradient(q_target) - q_selected
                td_error_weighted = self._td_error * self._importance_ph
                loss = tf.reduce_mean(tf.square(td_error_weighted), name='loss')
                self.opt = Optimizer.create(optimizer)
                self.opt.build(steps, self.global_step, self._obs_counter)
                self._train_op = self.opt.minimize(loss, self._weights)
        tensor_utils.add_observation_summary(self.net.input_ph, self.env)
        tf.summary.histogram('agent/action', action_onehot)
        tf.summary.histogram('agent/action_values', self.net.output)
        tf.summary.scalar('agent/learning_rate', self.opt.lr)
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

    def train(self, log_dir, render=False, test_render=False, test_episodes=1, callbacks=set()):
        """Starts training.

        Args:
            log_dir (str): Directory used for summary and checkpoints.
                Continues training, if checkpoint already exists.
            render (bool): Enables game screen rendering.
            test_render (bool): Enables rendering for test evaluations.
            test_episodes (int): Number of test episodes. To disable test evaluation, pass 0.
            callbacks (set): Set of AgentCallback instances.
        """
        try:
            self._train(log_dir, render, test_episodes, test_render, callbacks)
            logger.info('Training finished.')
        except KeyboardInterrupt:
            logger.info('Stopping training process...')
        if log_dir:
            self.save_weights(log_dir)

    def _train(self, log_dir, render, test_episodes, test_render, callbacks):
        reward_logger = tensor_utils.SummaryLogger(self.step_counter, self.obs_counter)
        writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        reward_stats = reinforceflow.utils.RewardStats()
        obs_counter = self.obs_counter
        last_log_time = time.time()
        last_target_sync = obs_counter
        obs = self.env.reset()
        while obs_counter < self._steps:
            obs_counter = self.increment_obs_counter()
            logs = {'obs_counter': obs_counter}
            [callback.on_iter_start(self, logs) for callback in callbacks]
            if render:
                self.env.render()
            action_values = self.predict_on_batch([obs])
            action = self._policy.select_action(self.env, action_values, obs_counter)
            obs_next, reward, term, info = self.env.step(action)
            logs['obs'] = obs
            logs['obs_next'] = obs_next
            logs['reward_raw'] = reward
            reward_stats.add(reward, term)
            self._exp.add(obs, action, reward, obs_next, term)
            obs = obs_next
            if self._exp.is_ready and obs_counter % self._update_freq == 0:
                b_obs, b_action, b_reward, b_obs_next, b_term, b_idxs, b_is = self._exp.sample()
                # self.ep_counter > last_log_ep and
                summarize = time.time() - last_log_time > self._log_freq
                td_error, summary_str = self.train_on_batch(b_obs, b_action, b_reward, b_obs_next,
                                                            b_term, summarize, b_is)
                if hasattr(self._exp, 'update'):
                    self._exp.update(b_idxs, np.abs(td_error))

                if obs_counter - last_target_sync > self._target_freq:
                    last_target_sync = obs_counter
                    self.target_update()

                if summarize:
                    self.save_weights(log_dir)
                    last_log_time = time.time()
                    self._async_eval(writer, reward_logger, test_episodes, test_render,
                                     train_stats=reward_stats)
                    [callback.on_log(self, logs) for callback in callbacks]
                    eps_log = [tf.Summary.Value(tag='agent/epsilon',
                                                simple_value=self._policy.epsilon)]
                    writer.add_summary(tf.Summary(value=eps_log), global_step=obs_counter)
                    if log_dir and summary_str:
                        writer.add_summary(summary_str, global_step=obs_counter)
            if term:
                self.increment_ep_counter()
                obs = self.env.reset()
            logs['term'] = term
            logs['action'] = action
            logs['reward'] = reward
            logs['episode_reward'] = reward_stats.episode_average()
            [callback.on_iter_end(self, logs) for callback in callbacks]
        logger.info('Performing final evaluation.')
        self._async_eval(writer, reward_logger, test_episodes, test_render)
        writer.close()
