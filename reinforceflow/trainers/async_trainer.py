from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from threading import Thread

import tensorflow as tf

from reinforceflow import logger
from reinforceflow.core import Stats
from reinforceflow.core.runner import EnvRunner
from reinforceflow.core.schedule import Schedule
from reinforceflow.core.stats import flush_stats
from reinforceflow.trainers.trainer import BaseTrainer
from reinforceflow.utils import tensor_utils


class AsyncTrainer(BaseTrainer):
    def __init__(self, agent, thread_agents, maxsteps, batch_size,
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
        self.thread_agents = thread_agents
        self.request_stop = False
        self.sync_ops = []
        self.thread_stats = []
        for th_agent in thread_agents:
            th_agent.sess = agent.sess
            with tf.device(th_agent.device), tf.variable_scope(th_agent.name):
                sync_op = [w.assign(self.agent.weights[i]) for i, w in enumerate(th_agent.weights)]
                self.sync_ops.append(sync_op)
        tensor_utils.initialize_variables(self.agent.sess)
        for th_agent in thread_agents:
            self.thread_stats.append(Stats(th_agent))

    def train_thread(self, thread_agent, sync_op, stats):
        runner = EnvRunner(agent=thread_agent, env=thread_agent.env, batch_size=self.batch_size,
                           sync_agent=self.agent)
        while self.agent.step < self.maxsteps:
            if self.request_stop:
                return
            self.agent.sess.run(sync_op)
            obs, action, reward, term, obs_next, traj_ends, infos = runner.sample()
            thread_agent.step = self.agent.step
            thread_agent.episode = self.agent.episode
            stats.add(action, reward, term, infos)
            thread_agent.train_on_batch(obs=obs,
                                        actions=action,
                                        rewards=reward,
                                        term=term,
                                        obs_next=obs_next,
                                        traj_ends=traj_ends,
                                        lr=self.lr_schedule.value(self.agent.step),
                                        summarize=False)

    def train(self):
        """Starts training."""
        writer = tf.summary.FileWriter(self.logdir, self.agent.sess.graph)
        threads = []
        for thread_agent, sync, stats in zip(self.thread_agents, self.sync_ops, self.thread_stats):
            thread_agent.sess = self.agent.sess
            t = Thread(target=self.train_thread, args=(thread_agent, sync, stats))
            t.daemon = True
            t.start()
            threads.append(t)
        self.request_stop = False
        last_log_time = time.time()
        try:
            while self.agent.step < self.maxsteps:
                if time.time() - last_log_time >= self.logfreq:
                    last_log_time = time.time()
                    flush_stats(self.thread_stats, name="%s Thread" % self.agent.name,
                                maxsteps=self.maxsteps, writer=writer)
                    self.agent.save_weights(self.logdir)
                    self.agent.test(self.test_env,
                                    self.test_episodes,
                                    max_steps=self.test_maxsteps,
                                    render=self.test_render,
                                    writer=writer)
                    writer.flush()
                if self.render:
                    [agent.env.render() for agent in self.thread_agents]
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
