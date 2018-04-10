from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from threading import Thread

import numpy as np
import tensorflow as tf

from reinforceflow import logger
from reinforceflow.core import Stats
from reinforceflow.core.runner import EnvRunner
from reinforceflow.core.schedule import Schedule
from reinforceflow.core.stats import flush_stats
from reinforceflow.trainers.trainer import BaseTrainer
from reinforceflow.utils import tensor_utils


class SyncTrainer(BaseTrainer):
    def __init__(self, agent, thread_envs, maxsteps, batch_size,
                 logdir, logfreq, log_on_term=True, lr_schedule=None,
                 render=False, test_env=None, test_render=False,
                 test_episodes=1, test_maxsteps=5e5):
        """Creates trainer based on Experience Replay buffer.

        Args:
            agent:
            maxsteps (int): Total amount of seen observations.
            logdir (str): Path used for summary and checkpoints.
            logfreq (int): Checkpoint and summary saving frequency (in seconds).
            log_on_term (bool): Whether to log only after episode ends.
            lr_schedule (core.Schedule): Learning rate scheduler.
            render (bool): Enables game screen rendering.
            test_env (gym.Env): Environment instance, used for testing.
            test_render (bool): Enables rendering for test evaluations.
            test_episodes (int): Number of test episodes. To disable test evaluation, pass 0.
            test_maxsteps (int): Maximum step allowed during test per episode.
        """
        self.agent = agent
        self.maxsteps = maxsteps
        self.batch_size = batch_size
        self.logdir = logdir
        self.logfreq = logfreq
        self.log_on_term = log_on_term
        self.render = render
        self.test_env = test_env
        self.test_render = test_render
        self.test_episodes = test_episodes
        self.test_maxsteps = test_maxsteps
        self.lr_schedule = Schedule.create(lr_schedule, agent.opt.lr, maxsteps)
        self._last_log_time = time.time()
        self._last_target_sync = self.agent.step
        self._summary_op = tf.summary.merge_all()
        self.thread_envs = thread_envs
        self.shared_batch = self._clear_batch()
        self.request_stop = False
        tensor_utils.initialize_variables(self.agent.sess)

    def _clear_batch(self):
        self.shared_batch = {'obs': [], 'actions': [], 'rewards': [], 'term': [],
                             'obs_next': [], 'traj_ends': [], 'thread_ready': []}
        return self.shared_batch

    def train_thread(self, uid, env, stats):
        runner = EnvRunner(agent=self.agent, env=env, batch_size=self.batch_size)
        while self.agent.step < self.maxsteps:
            if self.request_stop:
                return
            if uid in self.shared_batch['thread_ready']:
                continue
            obs, actions, rewards, terms, obs_next, traj_ends, infos = runner.sample()
            with self.agent.lock:
                stats.add(actions, rewards, terms, infos)
                self.shared_batch['obs'].extend(obs)
                self.shared_batch['actions'].extend(actions)
                self.shared_batch['rewards'].extend(rewards)
                self.shared_batch['term'].extend(terms)
                self.shared_batch['obs_next'].extend(obs_next)
                self.shared_batch['traj_ends'].extend(traj_ends)
                self.shared_batch['thread_ready'].append(uid)

    def train(self):
        """Starts training."""
        writer = tf.summary.FileWriter(self.logdir, self.agent.sess.graph)
        threads = []
        stats = []
        for uid, env in enumerate(self.thread_envs):
            stat = Stats(self.agent)
            stats.append(stat)
            t = Thread(target=self.train_thread, args=(uid, env, stat))
            t.daemon = True
            t.start()
            threads.append(t)
        self.request_stop = False
        last_log_time = time.time()
        try:
            while self.agent.step < self.maxsteps:
                # If shared batch is ready, perform gradient step
                if len(self.shared_batch['thread_ready']) >= len(self.thread_envs):
                    self.agent.train_on_batch(obs=np.asarray(self.shared_batch['obs']),
                                              actions=np.asarray(self.shared_batch['actions']),
                                              rewards=np.asarray(self.shared_batch['rewards']),
                                              term=np.asarray(self.shared_batch['term']),
                                              obs_next=np.asarray(self.shared_batch['obs_next']),
                                              traj_ends=np.asarray(self.shared_batch['traj_ends']),
                                              lr=self.lr_schedule.value(self.agent.step),
                                              summarize=False)
                    self.shared_batch = self._clear_batch()

                if time.time() - last_log_time >= self.logfreq:
                    last_log_time = time.time()
                    flush_stats(stats, name="%s Train" % self.agent.name,
                                maxsteps=self.maxsteps, writer=writer)
                    self.agent.save_weights(self.logdir)
                    self.agent.test(self.test_env,
                                    self.test_episodes,
                                    max_steps=self.test_maxsteps,
                                    render=self.test_render,
                                    writer=writer)
                    writer.flush()
                if self.render:
                    [env.render() for env in self.thread_envs]
                time.sleep(0.01)
        except KeyboardInterrupt:
            logger.info('Caught Ctrl+C! Stopping training process.')
        self.request_stop = True
        logger.info('Saving progress & performing evaluation.')
        self.agent.save_weights(self.logdir)
        self.agent.test(self.test_env, self.test_episodes, render=self.test_render)
        [t.join() for t in threads]
        logger.info('Training finished!')
        writer.close()

    def save(self):
        pass

    def load(self):
        pass
