from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import random
import threading

from six.moves import range  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

from reinforceflow.core.base_agent import BaseDQNAgent
from reinforceflow.core import EGreedyPolicy
from reinforceflow import misc
from reinforceflow import logger
from reinforceflow.misc import discount_rewards
from reinforceflow.envs.env_factory import make_env


class _GlobalDQNAgent(BaseDQNAgent):
    def __init__(self,
                 env,
                 net_fn,
                 log_dir,
                 name='GlobalAgent'):
        super(_GlobalDQNAgent, self).__init__(env=env, net_fn=net_fn, name=name)
        self._obs_step = tf.Variable(0, trainable=False)
        self._step_inc_op = self._obs_step.assign_add(1, use_locking=True)
        self.weights = self._weights
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.request_stop = False
        self.sess.run(tf.global_variables_initializer())
        self._prev_obs_step = self.obs_step
        self._prev_opt_step = self.optimizer_step
        self._last_time = time.time()

    @property
    def obs_step(self):
        return self.sess.run(self._obs_step)

    def step_increment(self):
        return self.sess.run(self._step_inc_op)

    def write_test_summary(self, test_episodes=3):
        test_r, test_q = self.test(episodes=test_episodes)
        obs_step = self.obs_step
        obs_per_sec = (self.obs_step - self._prev_obs_step) / (time.time() - self._last_time)
        opt_per_sec = (self.optimizer_step - self._prev_opt_step) / (time.time() - self._last_time)
        self._last_time = time.time()
        self._prev_obs_step = obs_step
        self._prev_opt_step = self.optimizer_step
        logger.info("Testing global agent: Average R: %.2f. Average maxQ: %.2f. Step: %d." % (test_r, test_q, obs_step))
        custom_values = [tf.Summary.Value(tag=self._scope_prefix + 'test_r', simple_value=test_r),
                         tf.Summary.Value(tag=self._scope_prefix + 'test_q', simple_value=test_q),
                         tf.Summary.Value(tag='observation/sec', simple_value=obs_per_sec),
                         tf.Summary.Value(tag='update/sec', simple_value=opt_per_sec)
                         ]
        self.writer.add_summary(tf.Summary(value=custom_values), global_step=obs_step)

    def train_on_batch(self, obs, actions, rewards, summarize=False):
        raise NotImplementedError('For training Async DQN, use AsyncDQNTrainer instead.')

    def _train(self, **kwargs):
        raise NotImplementedError('For training Async DQN, use AsyncDQNTrainer instead.')

    def train(self, **kwargs):
        raise NotImplementedError('For training Async DQN, use AsyncDQNTrainer instead.')


class _ThreadDQNAgent(BaseDQNAgent):
    def __init__(self,
                 env,
                 net_fn,
                 global_agent,
                 name=''):
        super(_ThreadDQNAgent, self).__init__(env=env, net_fn=net_fn, name=name)
        self.sess.close()
        self.sess = global_agent.sess
        self._sync_op = [self._weights[i].assign(global_agent.weights[i]) for i in range(len(self._weights))]
        self.global_agent = global_agent

    def prepare_train(self, optimizer, learning_rate, optimizer_args=None,
                      decay=None, decay_args=None, gradient_clip=40.0, saver_keep=10):
        super(_ThreadDQNAgent, self).prepare_train(optimizer, learning_rate, optimizer_args, decay, decay_args,
                                                   gradient_clip, saver_keep)
        self._grads_vars = list(zip(self._grads, self.global_agent.weights))
        self._train_op = self.global_agent.opt.apply_gradients(self._grads_vars)
        with tf.variable_scope(self._scope_prefix):
            for grad, w in self._grads_vars:
                tf.summary.histogram(w.name, w)
                tf.summary.histogram(w.name + '/gradients', grad)
            if len(self.env.observation_shape) == 1:
                tf.summary.histogram('observation', self._obs)
            elif len(self.env.observation_shape) <= 3:
                tf.summary.image('observation', self._obs)
            else:
                logger.warn('Cannot create summary for observation with shape %s' % self.env.obs_shape)
            tf.summary.histogram('action', self._action_one_hot)
            tf.summary.histogram('reward_per_action', self._q)
            tf.summary.scalar('learning_rate', self._lr)
            tf.summary.scalar('loss', self._loss)
            self._summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, self._scope_prefix))
        self.sess.run(tf.global_variables_initializer())

    def _synchronize(self):
        if self._sync_op is not None:
            self.sess.run(self._sync_op)

    def _train(self, **kwargs):
        pass

    def train(self,
              steps,
              optimizer,
              learning_rate,
              target_freq,
              policy,
              log_freq,
              optimizer_args=None,
              decay=None,
              decay_args=None,
              gradient_clip=40.0,
              gamma=0.99,
              batch_size=32,
              saver_keep=10):
        if not self._ready_for_train:
            self.prepare_train(optimizer, learning_rate, optimizer_args, decay, decay_args, gradient_clip, saver_keep)
        ep_reward = misc.IncrementalAverage()
        ep_q = misc.IncrementalAverage()
        reward_accum = 0
        last_log_step = 0
        episode = 0
        obs = self.env.reset()
        term = True
        while not self.global_agent.request_stop:
            self._synchronize()
            batch_obs, batch_rewards, batch_actions = [], [], []
            if term:
                term = False
                obs = self.env.reset()
            while not term and len(batch_obs) < batch_size:
                current_step = self.global_agent.step_increment()
                reward_per_action = self.predict(obs)
                batch_obs.append(obs)
                action = policy.select_action(self.env, reward_per_action, current_step)
                obs, reward, term, info = self.env.step(action)
                reward_accum += reward
                reward = np.clip(reward, -1, 1)
                batch_rewards.append(reward)
                batch_actions.append(action)
            expected_reward = 0
            if not term:
                # TODO: Clip expected reward?
                expected_reward = np.max(self.global_agent.target_predict(obs))
                ep_q.add(expected_reward)
            else:
                ep_reward.add(reward_accum)
                reward_accum = 0
            batch_rewards = discount_rewards(batch_rewards, gamma, expected_reward)
            summarize = term and log_freq and self.global_agent.obs_step - last_log_step > log_freq
            summary_str = self._train_on_batch(np.vstack(batch_obs), batch_actions, batch_rewards, summarize)
            if summarize:
                last_log_step = self.global_agent.obs_step
                train_r = ep_reward.reset()
                train_q = ep_q.reset()
                logger.info("%s - Train results: Average R: %.2f. Average maxQ: %.2f. Step: %d. Ep: %d"
                            % (self._scope_prefix, train_r, train_q, last_log_step, episode))
                if summary_str:
                    custom_values = [tf.Summary.Value(tag=self._scope_prefix + 'train_r', simple_value=train_r),
                                     tf.Summary.Value(tag=self._scope_prefix + 'train_q', simple_value=train_q),
                                     tf.Summary.Value(tag=self._scope_prefix + 'epsilon', simple_value=policy.epsilon),
                                     tf.Summary.Value(tag=self._scope_prefix + 'episodes', simple_value=episode)
                                     ]
                    self.global_agent.writer.add_summary(tf.Summary(value=custom_values), global_step=last_log_step)
                    self.global_agent.writer.add_summary(summary_str, global_step=last_log_step)


class AsyncDQNTrainer(object):
    def __init__(self, env, net_fn):
        self.env = env
        self.net_fn = net_fn

    def train(self,
              num_threads,
              steps,
              optimizer,
              learning_rate,
              log_dir,
              epsilon_steps,
              target_freq,
              log_freq,
              optimizer_args=None,
              gradient_clip=40.0,
              decay=None,
              decay_args=None,
              epsilon_pool=(0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.5, 0.5, 0.5),
              ckpt_dir=None,
              gamma=0.99,
              batch_size=32,
              render=False,
              saver_keep=10):
        if num_threads < 1:
            raise ValueError("Number of threads must be >= 1. Got: %s." % num_threads)
        threads = []
        envs = []
        if not isinstance(epsilon_pool, (list, tuple, np.ndarray)):
            epsilon_pool = list(epsilon_pool)
        global_agent = _GlobalDQNAgent(env=make_env(self.env),
                                       net_fn=self.net_fn,
                                       log_dir=log_dir,
                                       name='GlobalAgent')
        if ckpt_dir and tf.train.latest_checkpoint(log_dir) is not None:
            global_agent.load_weights(ckpt_dir)
        for t in range(num_threads):
            eps_min = random.choice(epsilon_pool)
            logger.debug("Sampling minimum epsilon = %0.2f for Thread-Agent #%d." % (eps_min, t))
            policy = EGreedyPolicy(eps_start=1.0, eps_final=eps_min, anneal_steps=epsilon_steps)
            env = make_env(self.env)
            envs.append(env)
            agent = _ThreadDQNAgent(env=env,
                                    net_fn=self.net_fn,
                                    global_agent=global_agent,
                                    name='ThreadAgent%d' % t)
            thread = threading.Thread(target=agent.train,
                                      args=(steps, optimizer, learning_rate, target_freq, policy, log_freq,
                                            optimizer_args, decay, decay_args, gradient_clip, gamma, batch_size,
                                            saver_keep))
            threads.append(thread)
        last_log_step = global_agent.obs_step
        last_target_update = last_log_step
        for t in threads:
            t.daemon = True
            t.start()
        global_agent.request_stop = False

        def has_live_threads():
            return True in [th.isAlive() for th in threads]

        while has_live_threads() and global_agent.obs_step < steps:
            try:
                if render:
                    for env in envs:
                        env.render()
                step = global_agent.obs_step
                if step - last_log_step >= log_freq:
                    last_log_step = step
                    global_agent.write_test_summary()
                    global_agent.save_weights(log_dir)

                if step - last_target_update >= target_freq:
                    last_target_update = step
                    global_agent.target_update()
                [t.join(1) for t in threads if t is not None and t.isAlive()]
                time.sleep(.01)
            except KeyboardInterrupt:
                logger.info('Caught Ctrl+C! Stopping training process.')
                global_agent.request_stop = True
                global_agent.save_weights(log_dir)
        logger.info('Training finished!')
