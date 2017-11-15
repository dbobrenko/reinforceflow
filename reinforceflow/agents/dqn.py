from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

from six.moves import range  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

import reinforceflow.utils
from reinforceflow.core.base_agent import BaseDQNAgent
from reinforceflow.core import Tuple
from reinforceflow import utils_tf
from reinforceflow import logger


class DQNAgent(BaseDQNAgent):
    def __init__(self, env, net_factory, use_double=True, use_gpu=True, name=''):
        """Constructs Deep Q-Network agent, based on paper:
        "Human-level control through deep reinforcement learning", Mnih et al., 2015.

        See `core.base_agent.BaseDQNAgent.__init__`.
        Args:
            use_double (bool): Enables Double DQN, described at:
                "Dueling Network Architectures for Deep Reinforcement Learning",
                Schaul et al., 2016.
        """
        super(DQNAgent, self).__init__(env=env, net_factory=net_factory, name=name)
        config = tf.ConfigProto(
            device_count={'GPU': use_gpu}
        )
        if isinstance(env.action_space, Tuple):
            raise ValueError("Current implementation of DQN doesn't supports Tuple action spaces.")
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self._use_double = use_double
        self._importance_ph = tf.placeholder('float32', [None], name='importance_sampling')
        self._term_ph = None
        self._td_error = None
        self._train_op = None
        self._target_net = None
        self._summary_op = None
        self._target_update = None

    def build_train_graph(self, optimizer, learning_rate, optimizer_args=None, gamma=0.99,
                          decay=None, decay_args=None, gradient_clip=40.0, saver_keep=10):
        """Builds training graph.

        Args:
            optimizer: An optimizer name string or class.
            learning_rate (float or Tensor): Optimizer's learning rate.
            optimizer_args (dict): Keyword arguments used for optimizer creation.
            gamma (float): Reward discount factor.
            decay (function): Learning rate decay.
                Expects tensorflow decay function or function name string.
                Valid: 'polynomial', 'exponential'. To disable, pass None.
            decay_args (dict): Keyword arguments, passed to the decay function.
            gradient_clip (float): Norm gradient clipping. To disable, pass 0 or None.
                To disable, pass False or None.
            saver_keep (int): Maximum number of checkpoints can be stored in `log_dir`.
                When exceeds, overwrites the most earliest checkpoints.
        """
        if self._train_op is not None:
            logger.warn("The training graph has already been built. Skipping.")
            return
        self._term_ph = tf.placeholder('float32', [None], name='term')
        with tf.variable_scope(self._scope + 'target_network') as scope:
            self._target_net = self._net_factory.make(input_space=self.env.obs_space,
                                                      output_space=self.env.action_space)
            target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope.name)
            self._target_update = [target_weights[i].assign(self._weights[i])
                                   for i in range(len(target_weights))]

        with tf.variable_scope(self._scope + 'optimizer'):
            opt, lr = utils_tf.create_optimizer(optimizer, learning_rate,
                                                optimizer_args=optimizer_args,
                                                decay=decay, decay_args=decay_args,
                                                global_step=self.global_step)
            action_onehot = tf.arg_max(self._action_ph, 1, name='action_argmax')
            action_onehot = tf.one_hot(action_onehot, self.env.action_space.shape[0],
                                       1.0, 0.0, name='action_one_hot')
            # Predict expected future reward for performed action
            q_selected = tf.reduce_sum(self.net.output * action_onehot, 1)
            if self._use_double:
                q_next_online_argmax = tf.arg_max(self.net.output, 1)
                q_next_online_onehot = tf.one_hot(q_next_online_argmax,
                                                  self.env.action_space.shape[0],
                                                  1.0)
                q_next_max = tf.reduce_sum(self._target_net.output * q_next_online_onehot, 1)
            else:
                q_next_max = tf.reduce_max(self._target_net.output, 1)
            q_next_max_masked = (1.0 - self._term_ph) * q_next_max
            q_target = self._reward_ph + gamma * q_next_max_masked
            self._td_error = tf.stop_gradient(q_target) - q_selected
            td_error_weighted = self._td_error * self._importance_ph
            loss = tf.reduce_mean(tf.square(td_error_weighted), name='loss')
            grads = tf.gradients(loss, self._weights)
            if gradient_clip:
                grads, _ = tf.clip_by_global_norm(grads, gradient_clip)
            grads_vars = list(zip(grads, self._weights))
            self._train_op = opt.apply_gradients(grads_vars,
                                                 global_step=self.global_step)
        save_vars = set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self._scope + 'network'))
        save_vars |= set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           self._scope + 'optimizer'))
        save_vars.add(self.global_step)
        save_vars.add(self._obs_counter)
        save_vars.add(self._ep_counter)
        self._saver = tf.train.Saver(var_list=list(save_vars), max_to_keep=saver_keep)
        utils_tf.add_grads_summary(grads_vars)
        utils_tf.add_observation_summary(self.net.input_ph, self.env)
        tf.summary.histogram('agent/action', action_onehot)
        tf.summary.histogram('agent/action_values', self.net.output)
        tf.summary.scalar('agent/learning_rate', lr)
        tf.summary.scalar('metrics/loss', loss)
        tf.summary.scalar('metrics/avg_Q', tf.reduce_mean(q_next_max))
        self._summary_op = tf.summary.merge_all()

    def _train(self, max_steps, update_freq, log_dir, render, target_freq, replay,
               policy, log_every_sec, test_episodes, test_render, ignore_checkpoint):
        avg_reward = reinforceflow.utils.IncrementalAverage()
        ep_reward = 0
        writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        if not ignore_checkpoint and log_dir and tf.train.latest_checkpoint(log_dir) is not None:
            self.load_weights(log_dir)
        obs = self.env.reset()
        reward_logger = utils_tf.SummaryLogger(self.step_counter, self.obs_counter)
        last_log_ep = self.ep_counter
        last_log_time = time.time()
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
                summarize = (self.ep_counter > last_log_ep
                             and time.time() - last_log_time > log_every_sec)
                td_error, summary_str = self.train_on_batch(b_obs, b_action, b_reward, b_obs_next,
                                                            b_term, summarize, b_importances)
                if hasattr(replay, 'update'):
                    replay.update(b_idxs, np.abs(td_error))

                if step % target_freq == target_freq-1:
                    self.target_update()

                if summarize:
                    self.save_weights(log_dir)
                    last_log_time = time.time()
                    last_log_ep = self.ep_counter
                    step = self.step_counter
                    if test_episodes:
                        test_rewards = self.test(episodes=test_episodes,
                                                 render=test_render, copy_env=True)
                        reward_summary = reward_logger.summarize(avg_reward, test_rewards,
                                                                 self.ep_counter, step, obs_counter)
                        writer.add_summary(reward_summary, global_step=step)
                    eps_log = [tf.Summary.Value(tag='agent/epsilon', simple_value=policy.epsilon)]
                    writer.add_summary(tf.Summary(value=eps_log), global_step=step)
                    if log_dir and summary_str:
                        writer.add_summary(summary_str, global_step=step)
            if term:
                self.increment_ep_counter()
                avg_reward.add(ep_reward)
                ep_reward = 0
                obs = self.env.reset()
        writer.close()

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

    def train(self,
              max_steps,
              optimizer,
              learning_rate,
              log_dir,
              replay,
              policy,
              target_freq,
              update_freq=1,
              optimizer_args=None,
              decay=None,
              decay_args=None,
              gradient_clip=40.0,
              render=False,
              gamma=0.99,
              log_every_sec=600,
              saver_keep=3,
              ignore_checkpoint=False,
              test_render=False,
              test_episodes=3,
              **kwargs):
        """Starts training process.

        Args:
            max_steps (int): Number of training steps (optimizer steps).
            optimizer: An optimizer string name or class.
            learning_rate (float or Tensor): Optimizer learning rate.
            log_dir (str): Directory used for summary and checkpoints.
                Continues training, if checkpoint already exists.
            replay: (core.ExperienceReplay) Experience buffer.
            policy (core.BasePolicy): Agent's training policy.
            target_freq (int): Target network update frequency (in update steps).
            update_freq (int): Optimizer update frequency.
            optimizer_args (dict): Keyword arguments used for optimizer creation.
            decay (function): Learning rate decay.
                Expects tensorflow decay function or function name string.
                Available names: 'polynomial', 'exponential'.
                To disable, pass None.
            decay_args (dict): Keyword arguments used for learning rate decay function creation.
            gradient_clip (float): Norm gradient clipping. To disable, pass 0 or None.
            render (bool): Enables game screen rendering.
            gamma (float): Reward discount factor.
            log_every_sec (int): Checkpoint and summary saving frequency (in seconds).
            saver_keep (int): Maximum number of checkpoints can be stored in `log_dir`.
                When exceeds, overwrites the most earliest checkpoints.
            ignore_checkpoint (bool): If enabled, training will start from scratch,
                and overwrite all old checkpoints found at `log_dir` path.
            test_render (bool): Enables rendering for test evaluations.
            test_episodes (int): Number of test episodes. To disable test evaluation, pass 0.
        """
        self.build_train_graph(optimizer, learning_rate, optimizer_args, gamma,
                               decay, decay_args, gradient_clip, saver_keep)
        try:
            self._train(max_steps, update_freq, log_dir, render, target_freq, replay,
                        policy, log_every_sec, test_episodes, test_render, ignore_checkpoint)
            logger.info('Training finished.')
        except KeyboardInterrupt:
            logger.info('Stopping training process...')
        if log_dir:
            self.save_weights(log_dir)
