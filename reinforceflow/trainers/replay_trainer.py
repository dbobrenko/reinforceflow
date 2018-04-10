from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from threading import Thread

import numpy as np
import tensorflow as tf

from reinforceflow import logger
from reinforceflow.core import ProportionalReplay, Stats, flush_stats
from reinforceflow.core.runner import ReplayRunner
from reinforceflow.core.schedule import Schedule
from reinforceflow.trainers.trainer import BaseTrainer


class ReplayTrainer(BaseTrainer):
    def __init__(self, env, agent, maxsteps, replay,
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
            replay (core.ExperienceReplay): Experience replay buffer.
            render (bool): Enables game screen rendering.
            test_env (gym.Env): Environment instance, used for testing.
            test_render (bool): Enables rendering for test evaluations.
            test_episodes (int): Number of test episodes. To disable test evaluation, pass 0.
            test_maxsteps (int): Maximum step allowed during test per episode.
        """
        self.agent = agent
        self.maxsteps = maxsteps
        self.replay = replay
        self.logdir = logdir
        self.logfreq = logfreq
        self.log_on_term = log_on_term
        self.lr_schedule = lr_schedule
        self.render = render
        self.test_env = test_env
        self.test_render = test_render
        self.test_episodes = test_episodes
        self.test_maxsteps = test_maxsteps
        self.runner = ReplayRunner(agent=agent, env=env, replay=replay)
        self.train_stats = Stats(self.agent)
        self.perform_stats = Stats(self.agent)
        self._last_log_time = time.time()
        self._last_target_sync = self.agent.step
        self._summary_op = tf.summary.merge_all()

    @staticmethod
    def collect_sample(obs, agent, replay, stats):
        action = agent.explore(obs, agent.step)
        obs_next, reward, term, info = agent.env.step(action)
        stats.add(action, rewards=reward, terms=term, infos=info)
        replay.add(obs, action, reward, term, obs_next)
        obs = obs_next
        if term:
            obs = agent.env.reset()
        return obs

    @staticmethod
    def collect_replay(maxsteps, agent, replay, stats, render):
        obs = agent.env.reset()
        while agent.step < maxsteps:
            if render:
                agent.env.render()
            obs = ReplayTrainer.collect_sample(obs, agent, replay, stats)

    def train(self):
        """Starts training."""
        try:
            lr_schedule = Schedule.create(self.lr_schedule, self.agent.opt.lr,
                                          self.maxsteps)
            writer = tf.summary.FileWriter(self.logdir, self.agent.sess.graph)
            t = Thread(target=self.collect_replay, args=(self.maxsteps, self.agent, self.replay,
                                                         self.train_stats, self.render))

            t.daemon = True
            t.start()
            while self.agent.step < self.maxsteps:
                if not self.replay.is_ready:
                    continue

                obs, actions, rewards, term, obs_next, ends, idxs, importance = self.runner.sample()
                # TODO info and lr (take from train on batch dict?)
                lr = lr_schedule.value(self.agent.step)
                self.perform_stats.add(actions, rewards, term, {})
                summarize = time.time() - self._last_log_time > self.logfreq
                res = self.agent.train_on_batch(obs=obs,
                                                actions=actions,
                                                rewards=rewards,
                                                term=term,
                                                obs_next=obs_next,
                                                traj_ends=ends,
                                                lr=lr,
                                                summarize=summarize,
                                                importance=importance)

                if isinstance(self.replay, ProportionalReplay):
                    # TODO value methods
                    self.replay.update(idxs,
                                       np.abs(np.sum(res['value'] * actions, 1) - res['target']))

                if summarize:
                    self._last_log_time = time.time()
                    self.agent.save_weights(self.logdir)
                    flush_stats(self.perform_stats, "%s Performance" % self.agent.name,
                                log_progress=False, log_rewards=False, log_hyperparams=False,
                                writer=writer)
                    flush_stats(self.train_stats, "%s Train" % self.agent.name,
                                log_performance=False, log_hyperparams=False,
                                maxsteps=self.maxsteps, writer=writer)
                    self.agent.test(self.test_env,
                                    self.test_episodes,
                                    max_steps=self.test_maxsteps,
                                    render=self.test_render,
                                    writer=writer)
                    if self.logdir and 'summary' in res:
                        writer.add_summary(res['summary'], global_step=self.agent.step)
                    writer.flush()

            logger.info('Performing final evaluation.')
            self.agent.test(self.test_env,
                            self.test_episodes,
                            max_steps=self.test_maxsteps,
                            render=self.test_render)
            writer.close()
            logger.info('Training finished.')
        except KeyboardInterrupt:
            logger.info('Stopping training process...')
        self.agent.save_weights(self.logdir)

    def save(self):
        pass

    def load(self):
        pass
