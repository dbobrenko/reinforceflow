from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import threading
import time
from threading import Thread

import tensorflow as tf
from six.moves import range  # pylint: disable=redefined-builtin

from reinforceflow import logger
from reinforceflow.core import BasePolicy, Stats
from reinforceflow.core.agent import BaseAgent
from reinforceflow.core.optimizer import Optimizer
from reinforceflow.core.schedule import Schedule
from reinforceflow.core.stats import ThreadStats


class AsyncAgent(BaseAgent):
    def __init__(self, env, model, thread_agent, optimizer, num_threads, is_training=True,
                 device='/gpu:0', saver_keep=3, name=''):
        """Base class for asynchronous agents, based on paper:
        "Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016.
        (https://arxiv.org/abs/1602.01783v2)

        See `BaseAgent`.

        Args:
            env (gym.Env): Environment instance.
            model (models.Model): Model.
            optimizer (str or Optimizer): [Training-only] Agent's optimizer.
                By default: RMSProp(lr=7e-4, epsilon=0.1, decay=0.99, lrdecay='linear').
            num_threads (int): [Training-only] Amount of asynchronous threads for training.
            is_training (bool): Builds train graph, if enabled.
            device (str): TensorFlow device, used for graph creation.
            saver_keep (int): [Training-only] Maximum number of checkpoints can be stored at once.
        """
        super(AsyncAgent, self).__init__(env=env, model=model, name=name)
        self.lock = threading.Lock()

        if not is_training:
            return

        with tf.device(self.device):
            with tf.variable_scope(self._scope + 'optimizer') as sc:
                self.opt = Optimizer.create(optimizer)
                self.opt.build(self.global_step)
                self._savings |= set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, sc.name))
        self._saver = tf.train.Saver(var_list=list(self._savings), max_to_keep=saver_keep)
        self.weights = self._weights

        self.request_stop = False
        if num_threads < 1:
            raise ValueError("Number of threads must be >= 1 (Got: %s)." % num_threads)
        self._num_threads = num_threads
        self._thread_agents = []
        for t in range(num_threads):
            agent = thread_agent(env=copy.deepcopy(self.env),
                                 model=self.model,
                                 global_agent=self,
                                 device=device,
                                 name='ThreadAgent%d' % t)
            self._thread_agents.append(agent)

    def train(self, maxsteps, policy, log_dir, log_freq, batch_size=5, gamma=0.99, lr_schedule=None,
              log_on_term=True, render=False, test_env=None, test_render=False, test_episodes=1,
              test_maxsteps=5000, callbacks=set()):
        """Starts training.

        Args:
            maxsteps (int): Total amount of seen observations across all threads.
            policy (core.BasePolicy): Agent's training policy.
            log_dir (str): Path used for summary and checkpoints.
            log_freq (int): Checkpoint and summary saving frequency (in seconds).
            batch_size (int): Training batch size.
            gamma (float): Reward discount factor.
            lr_schedule (Schedule or str): Learning rate schedule.
            log_on_term (bool): Whether to log only after episode ends.
            render (bool): Enables game screen rendering.
            test_env (gym.Env): Environment instance, used for testing.
            test_render (bool): Enables rendering for test evaluations.
            test_episodes (int): Number of test episodes. To disable test evaluation, pass 0.
            test_maxsteps (int): Maximum steps allowed during test per episode.
            callbacks (set): Set of AgentCallback instances.
        """
        lr_schedule = Schedule.create(lr_schedule, self.opt.learning_rate, maxsteps)
        writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        if isinstance(policy, BasePolicy):
            policy = [copy.deepcopy(policy) for _ in range(self._num_threads)]
        if len(policy) != self._num_threads:
            raise ValueError("Amount of policies must be equal to the amount of threads.")
        threads = []
        stats = ThreadStats([], 'Average Stats', file_writer=writer, initial_step=self.step)
        for i, agent in enumerate(self._thread_agents):
            local_stats = Stats(log_freq=None)
            t = Thread(target=agent.train, kwargs={'log_freq': log_freq,
                                                   'policy': policy[i],
                                                   'gamma': gamma,
                                                   'batch_size': batch_size,
                                                   'lr_schedule': lr_schedule,
                                                   'stats': local_stats,
                                                   'writer': writer})
            t.daemon = True
            t.start()
            threads.append(t)
            stats.thread_stats.append(local_stats)
        self.request_stop = False
        last_log_time = time.time()
        try:
            while self.step < maxsteps:
                [callback.on_iter_start(self, {}) for callback in callbacks]
                if time.time() - last_log_time >= log_freq:
                    last_log_time = time.time()
                    self.save_weights(log_dir)
                    stats.flush(self.step, self.episode)
                    self.test(test_env, test_episodes, max_steps=test_maxsteps, render=test_render,
                              writer=writer)
                    [callback.on_log(self, {}) for callback in callbacks]
                if render:
                    [agent.env.render() for agent in self._thread_agents]
                [callback.on_iter_end(self, {}) for callback in callbacks]
                time.sleep(0.01)
        except KeyboardInterrupt:
            logger.info('Caught Ctrl+C! Stopping training process.')
        self.request_stop = True
        logger.info('Saving progress & performing evaluation.')
        self.save_weights(log_dir)
        self.test(test_env, test_episodes, render=test_render)
        [t.join() for t in threads]
        logger.info('Training finished!')
        writer.close()

    def train_on_batch(self, *args, **kwargs):
        raise ValueError("Training on batch is not supported. Use `train` method instead.")


class AsyncThreadAgent(BaseAgent):
    def __init__(self, env, model, global_agent, device, name=''):
        super(AsyncThreadAgent, self).__init__(env=env, model=model, name=name, device=device)
        self.sess = global_agent.sess
        self.global_agent = global_agent

    def train_on_batch(self, obs, actions, rewards, obs_next, term, lr,
                       gamma=0.99, summarize=False):
        raise NotImplementedError

    def _sync_global(self):
        self.global_agent.sess.run(self._sync_op)

    def train(self, log_freq, policy, gamma, batch_size, lr_schedule, stats, writer=None):
        last_log_t = time.time()
        obs = self.env.reset()
        term = True
        while not self.global_agent.request_stop:
            self._sync_global()
            batch_obs, batch_rewards, batch_actions = [], [], []
            if term:
                term = False
                obs = self.env.reset()
            while not term and len(batch_obs) < batch_size:
                batch_obs.append(obs)
                action = self.predict_action(obs, policy, self.global_agent.step)
                obs, reward, term, info = self.env.step(action)
                with self.global_agent.lock:
                    self.global_agent.step += 1
                    self.global_agent.episode += int(term)
                stats.add(reward, term, info, step=self.global_agent.step,
                          episode=self.global_agent.episode)
                batch_rewards.append(reward)
                batch_actions.append(action)
            write_summary = log_freq and time.time() - last_log_t > log_freq
            summary_str = self.train_on_batch(batch_obs, batch_actions, batch_rewards, [obs],
                                              term, lr=lr_schedule.value(self.global_agent.step),
                                              gamma=gamma,
                                              summarize=write_summary)
            if write_summary:
                last_log_t = time.time()
                logs = [tf.Summary.Value(tag=self._scope + 'epsilon', simple_value=policy.epsilon)]
                writer.add_summary(tf.Summary(value=logs), global_step=self.global_agent.step)
                if summary_str:
                    writer.add_summary(summary_str, global_step=self.global_agent.step)
                # writer.flush()

    def close(self):
        pass

