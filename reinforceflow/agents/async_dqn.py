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
                 optimizer,
                 learning_rate,
                 net_fn,
                 log_dir,
                 optimizer_args=None,
                 decay=None,
                 decay_args=None,
                 gradient_clip=40.0,
                 name='GlobalAgent'):
        super(_GlobalDQNAgent, self).__init__(env=env, net_fn=net_fn, optimizer=optimizer, learning_rate=learning_rate,
                                              optimizer_args=optimizer_args, gradient_clip=gradient_clip, decay=decay,
                                              decay_args=decay_args, name=name)
        self._step_inc_op = self.global_step.assign_add(1, use_locking=True)
        self.weights = self._weights
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.request_stop = False
        self.sess.run(tf.global_variables_initializer())

    def step_increment(self):
        return self.sess.run(self._step_inc_op)

    def write_test_summary(self):
        test_r, test_q = self.test(episodes=3)
        step = self.current_step
        logger.info("Testing global agent: Average R: %.2f. Average maxQ: %.2f. Step: %d." % (test_r, test_q, step))
        custom_values = [tf.Summary.Value(tag=self._scope_prefix + 'test_r', simple_value=test_r),
                         tf.Summary.Value(tag=self._scope_prefix + 'test_q', simple_value=test_q),
                         ]
        self.writer.add_summary(tf.Summary(value=custom_values), global_step=step)

    def _train(self, **kwargs):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError


class _ThreadDQNAgent(BaseDQNAgent):
    def __init__(self,
                 env,
                 optimizer,
                 learning_rate,
                 net_fn,
                 global_agent,
                 optimizer_args=None,
                 decay=None,
                 decay_args=None,
                 gradient_clip=40.0,
                 name=''):
        super(_ThreadDQNAgent, self).__init__(env=env, net_fn=net_fn, optimizer=optimizer, learning_rate=learning_rate,
                                              optimizer_args=optimizer_args, gradient_clip=gradient_clip, decay=decay,
                                              decay_args=decay_args, name=name)
        self._grads_vars = list(zip(self._grads, global_agent.weights))
        self._train_op = global_agent.opt.apply_gradients(self._grads_vars)
        self._sync_op = [self._weights[i].assign(global_agent.weights[i]) for i in range(len(self._weights))]
        self.global_agent = global_agent
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
              max_steps,
              target_freq=10000,
              gamma=0.99,
              policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=1000000),
              log_freq=10000,
              batch_size=32):
        ep_reward = misc.IncrementalAverage()
        ep_q = misc.IncrementalAverage()
        reward_accum = 0
        last_log_step = 0
        episode = 0
        obs = self.env.reset()
        last_time = time.time()
        prev_log_step = self.global_agent.current_step
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
                expected_reward = np.max(self.global_agent.predict_target(obs))
                ep_q.add(expected_reward)
            else:
                ep_reward.add(reward_accum)
                reward_accum = 0
            batch_rewards = discount_rewards(batch_rewards, gamma, expected_reward)
            summarize = term and log_freq and self.global_agent.current_step - last_log_step > log_freq
            summary_str = self.train_on_batch(np.vstack(batch_obs), batch_actions, batch_rewards, summarize)
            if summarize:
                last_log_step = self.global_agent.current_step
                train_r = ep_reward.reset()
                train_q = ep_q.reset()
                logger.info("%s - Train results: Average R: %.2f. Average maxQ: %.2f. Step: %d. Ep: %d"
                            % (self._scope_prefix, train_r, train_q, last_log_step, episode))
                if summary_str:
                    step_per_sec = (last_log_step - prev_log_step) / (time.time() - last_time)
                    last_time = time.time()
                    prev_log_step = last_log_step
                    custom_values = [tf.Summary.Value(tag=self._scope_prefix + 'train_r', simple_value=train_r),
                                     tf.Summary.Value(tag=self._scope_prefix + 'train_q', simple_value=train_q),
                                     tf.Summary.Value(tag=self._scope_prefix + 'epsilon', simple_value=policy.epsilon),
                                     tf.Summary.Value(tag=self._scope_prefix + 'step/sec', simple_value=step_per_sec)
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
              epsilon_pool=None,
              ckpt_dir=None,
              gamma=0.99,
              batch_size=32,
              render=False):
        if num_threads < 1:
            raise ValueError("Number of threads must be >= 1. Got: %s." % num_threads)
        threads = []
        envs = []
        if epsilon_pool is None:
            epsilon_pool = 4*[0.1] + 3*[0.01] + 3*[0.5]
        if not isinstance(epsilon_pool, (list, tuple, np.ndarray)):
            epsilon_pool = list(epsilon_pool)
        global_agent = _GlobalDQNAgent(env=make_env(self.env),
                                       optimizer=optimizer,
                                       learning_rate=learning_rate,
                                       net_fn=self.net_fn,
                                       optimizer_args=optimizer_args,
                                       decay=decay,
                                       decay_args=decay_args,
                                       gradient_clip=gradient_clip,
                                       log_dir=log_dir,
                                       name='GlobalAgent')
        if ckpt_dir:
            global_agent.load_weights(ckpt_dir)
        for t in range(num_threads):
            eps_min = random.choice(epsilon_pool)
            logger.debug("Sampling minimum epsilon = %0.2f for Thread-Agent #%d." % (eps_min, t))
            policy = EGreedyPolicy(eps_start=1.0, eps_final=eps_min, anneal_steps=epsilon_steps)
            env = make_env(self.env)
            envs.append(env)
            agent = _ThreadDQNAgent(env=env,
                                    optimizer=optimizer,
                                    learning_rate=learning_rate,
                                    net_fn=self.net_fn,
                                    optimizer_args=optimizer_args,
                                    decay=decay,
                                    decay_args=decay_args,
                                    gradient_clip=gradient_clip,
                                    global_agent=global_agent,
                                    name='ThreadAgent%d' % t)
            thread = threading.Thread(target=agent.train,
                                      args=(steps, target_freq, gamma, policy, log_freq, batch_size))
            threads.append(thread)
        last_log_step = global_agent.current_step
        last_target_update = last_log_step
        for t in threads:
            t.daemon = True
            t.start()
        global_agent.request_stop = False

        def has_live_threads():
            return True in [th.isAlive() for th in threads]

        while has_live_threads() and global_agent.current_step < steps:
            try:
                if render:
                    for env in envs:
                        env.render()
                step = global_agent.current_step
                if step - last_log_step >= log_freq:
                    last_log_step = step
                    global_agent.write_test_summary()
                    global_agent.save_weights(log_dir)

                if step - last_target_update >= target_freq:
                    last_target_update = step
                    global_agent.update_target()
                [t.join(1) for t in threads if t is not None and t.isAlive()]
                time.sleep(.01)
            except KeyboardInterrupt:
                logger.info('Caught Ctrl+C! Stopping training process.')
                global_agent.request_stop = True
                global_agent.save_weights(log_dir)
        logger.info('Training finished!')
