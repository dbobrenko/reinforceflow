from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import os
import random

from six.moves import range  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

from reinforceflow.core.base_agent import BaseDQNAgent, BaseDiscreteAgent
from reinforceflow.agents.dqn import DQNAgent
from reinforceflow.core import ExperienceReplay
from reinforceflow.nets import dqn
from reinforceflow.core import EGreedyPolicy, GreedyPolicy
from reinforceflow import misc
from reinforceflow import logger


class AsyncDQNAgent(BaseDQNAgent):
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
        super(AsyncDQNAgent, self).__init__(env, sess, net_fn, gradient_clip, opt, learning_rate, decay,
                                            decay_poly_end_lr, decay_poly_power, decay_poly_steps, decay_rate)
        with tf.variable_scope('target_update'):
            target_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_network')
            w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'network')
            self._target_update = [target_w[i].assign(w[i]) for i in range(len(target_w))]

    def train(self, **kwargs):
        raise NotImplementedError


class AsyncDQNTrainer(BaseDiscreteAgent):
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
        super(AsyncDQNTrainer, self).__init__(env=env)
        eps_min = random.choice(EPS_MIN_SAMPLES)
        print('Thread: %d. Sampled min epsilon: %f' % (thread_idx, eps_min))
        self._training_finished = False

    # def predict(self, obs):
    #     return self.sess.run(self._q, {self._obs: obs})
    #
    # def predict_target(self, obs):
    #     return self.sess.run(self._target_q, {self._target_obs: obs})
    #
    # def update_target(self):
    #     self.sess.run(self._target_update)
    #
    # def save_weights(self, path, model_name='model.ckpt'):
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     self._saver.save(self.sess, os.path.join(path, model_name), global_step=self.global_step)
    #     logger.info('Checkpoint saved to %s' % os.path.join(path, model_name))
    #
    # def load_weights(self, checkpoint):
    #     if not os.path.exists(checkpoint):
    #         raise ValueError('Checkpoint path/dir %s does not exists.' % checkpoint)
    #     if tf.gfile.IsDirectory(checkpoint):
    #         checkpoint = tf.train.latest_checkpoint(checkpoint)
    #     logger.info('Restoring checkpoint from %s', checkpoint)
    #     self._saver.restore(self.sess, save_path=checkpoint)
    #     self.update_target()

    def _worker(self, agent, policy, gamma, total_frames, tmax, update_freq, log_freq, logdir, thread_idx=0):
        last_logging = agent.frame
        last_target_update = agent.frame
        terminal = True
        obs = None

        while agent.frame < total_frames:
            batch_states, batch_rewards, batch_actions = [], [], []
            if terminal:
                terminal = False
                obs = self.env.reset_random()
            batch = []
            while not terminal and len(batch_states) < tmax:
                step = agent.step
                # Increment shared frame counter
                agent.frame_increment()
                batch_states.append(obs)
                reward_per_action = agent.predict(obs)
                action = policy.select_action(reward_per_action, self.env, step)
                obs_next, reward, term, info = self.env.step(action)
                reward = np.clip(reward, -1, 1)
                # one-step Q-Learning: add discounted expected future reward
                if not terminal:
                    reward += gamma * agent.predict_target(obs)
                batch.append({'obs': obs, 'action': action, 'reward': reward, 'obs_next': obs_next, 'term': term})
                obs = obs_next
            # Apply asynchronous gradient update to shared agent
            agent.train_on_batch(batch)
            # Logging and target network update
            if thread_idx == 0:
                if agent.frame - last_target_update >= update_freq:
                    last_target_update = agent.step
                    agent.update_target()
                if agent.frame - last_logging >= log_freq and terminal:
                    pass
        self._training_finished = True
        print('Thread %d. Training finished. Total frames: %s' % (thread_idx, agent.frame))

    def train(self, threads, logdir, render=False):
        """Launches worker asynchronously in 'FLAGS.threads' threads
        :param worker: worker function"""
        print('Starting %s threads.' % threads)
        self._training_finished = False
        processes = []
        envs = []
        agents = []
        for _ in range(threads):
            # TODO: recreate
            envs.append(self.env.recreate())

        with tf.Session() as sess:
            agent = BaseDQNAgent(env=sess)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
            sess.run(tf.global_variables_initializer())
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            ckpt = tf.train.latest_checkpoint(logdir)
            if ckpt is not None:
                saver.restore(sess, ckpt)
                agent.update_target()
                print('Restoring session from %s' % ckpt)
            # for i in range(threads):
                # processes.append(th.Thread(target=self._worker, args=(agent, policy, gamma, total_frames, tmax, update_freq, log_freq, logdir, 0)))
            for p in processes:
                p.daemon = True
                p.start()
            while not self._training_finished:
                if render:
                    for i in range(threads):
                        envs[i].render()
                time.sleep(.01)
            for p in processes:
                p.join()
