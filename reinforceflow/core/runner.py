from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin


class Runner(object):
    def __init__(self, agent, env, batch_size, replay=None):
        """Adapter for batch sampling from environments and replays.

        Args:
            agent (Agent): Learning agent.
            env (gym.Env): Environment with gym-like interface.
            batch_size (int): Batch size.
            replay (Replay): Replay for training. To disable training from replay, pass None.
        """
        if replay is None:
            self.runner = EnvRunner(agent, env, batch_size)
        else:
            self.runner = ReplayRunner(agent, env, batch_size, replay=replay)

    def sample(self):
        """Samples batch from given data provider.
        Increments agent's step and episode counters.

        Returns:
            Batch: (observation, reward, terminal, info).
        """
        return self.runner.sample()


class EnvRunner(object):
    def __init__(self, agent, env, batch_size, sample_on_term=True, sync_agent=None):
        """Wrapper for environment batch sampling.

        Args:
            agent (Agent): Learning agent.
            env (gym.Env): Environment with gym-like interface.
            batch_size (int): Batch size.
        """
        self.agent = agent
        self.env = env
        self.batch_size = batch_size
        self._obs = None
        self.sample_on_term = sample_on_term
        self.sync_agent = agent if sync_agent is None else sync_agent

    def sample(self):
        """Acts Samples batch from given environment.
        Increments agent's step and episode counters.

        Returns:
            Batch with replay: (observation, reward, terminal).
        """
        obs = []
        acts = []
        rewards = []
        obses_next = []
        terms = []
        trajends = []
        infos = []
        if self._obs is None:
            self._obs = self.env.reset()
        for i in range(self.batch_size):
            act = self.agent.explore(self._obs)
            obs_next, reward, term, info = self.env.step(act)
            obs.append(self._obs)
            acts.append(act)
            rewards.append(reward)
            terms.append(term)
            obses_next.append(obs_next)
            trajends.append(term)
            infos.append(info)
            if term:
                self._obs = self.env.reset()
                if self.sample_on_term:
                    break
            self._obs = obs_next
        trajends[-1] = True
        with self.sync_agent.lock:
            self.sync_agent.step += self.batch_size
            self.sync_agent.episode += sum(terms)
        return (np.asarray(obs), np.asarray(acts), np.asarray(rewards),
                np.asarray(terms), np.asarray(obses_next),
                np.asarray(trajends), infos)


class ReplayRunner(object):
    def __init__(self, agent, env, replay, shuffle=False):
        """Adapter for batch sampling from environments and replays.

        Args:
            agent (Agent): Learning agent.
            env (gym.Env): Environment with gym-like interface.
            replay (Replay): Replay for training. To disable training from replay, pass None.
        """
        self.agent = agent
        self.env = env
        self.replay = replay
        self.replay.shuffle = shuffle
        self._obs = None

    def sample(self):
        """Samples batch from given data provider.
        Increments agent's step and episode counters.

        Returns:
            Batch with replay: (obs, action, reward, terminal, next-obs).
        """
        batch = self.replay.sample()
        # Count terminal states
        self.agent.episode += sum(batch[3])
        self.agent.step += len(batch[0])
        return batch
