from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from operator import itemgetter
import random
import numpy as np
from reinforceflow.core.datastruct import SumTree, MinTree
from reinforceflow import logger


class ExperienceReplay(object):
    """Experience replay buffer.

    Args:
        capacity (int):  Total replay capacity.
        batch_size (int): Size of sampled batch.
        min_size (int): Minimum replay size (enables is_ready property, when fills).
    """
    def __init__(self, capacity, batch_size, min_size=0):
        if batch_size < 1:
            raise ValueError("Batch size must be higher or equal to 1.")
        if capacity < batch_size:
            logger.warn("Minimum capacity must be higher or equal "
                        "to the batch size (Got: %s). "
                        "Setting minimum buffer size to the batch size." % capacity)
            capacity = batch_size
        self._capacity = capacity
        self._batch_size = batch_size
        self._min_size = max(batch_size, min_size)
        # Python lists offers ~18% faster index access speed at current setup,
        # at the same time sacrificing ~18% of memory compared to numpy.ndarray.
        self._obs = [0] * (capacity + 1)
        self._actions = [0] * capacity
        self._rewards = [0] * capacity
        self._terms = [0] * capacity
        self._idx = 0
        self._size = 0

    def _cycle_idx(self, idx):
        return idx % self._capacity

    def add(self, obs, action, reward, obs_next, term):
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._terms[self._idx] = term
        self._obs[self._idx] = obs
        self._obs[self._idx + 1] = obs if term else obs_next
        self._idx = self._cycle_idx(self._idx + 1)
        self._size = min(self._size + 1, self._capacity)

    def sample(self):
        rand_idxs = random.sample(range(self._size), self._batch_size)
        gather = itemgetter(*rand_idxs)
        next_obs_gather = itemgetter(*[i + 1 for i in rand_idxs])
        return (gather(self._obs),
                gather(self._actions),
                gather(self._rewards),
                next_obs_gather(self._obs),
                gather(self._terms),
                rand_idxs,
                [1.0] * len(rand_idxs))

    @property
    def size(self):
        return self._size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def is_ready(self):
        return self._size >= self._min_size

    def __len__(self):
        return self._size


class ProportionalReplay(ExperienceReplay):
    """Proportional Prioritized Experience replay buffer.
    Based on paper: https://arxiv.org/pdf/1511.05952.pdf
    Args:
        capacity (int):  Total replay capacity.
        batch_size (int): Size of sampled batch.
        min_size (int): Minimum replay size (enables is_ready property, when fills).
        alpha (float): Exponent which determines how much priority is used.
            (0 - uniform prioritization, 1 - full prioritization).
        beta (float): Exponent which determines how much importance-sampling correction is used.
            (0 - no correction, 1 - full correction).
    """
    def __init__(self, capacity, batch_size, min_size=0, alpha=0.7, beta=0.5):
        super(ProportionalReplay, self).__init__(capacity, batch_size, min_size)
        assert alpha >= 0
        assert beta >= 0
        self.sumtree = SumTree(capacity)
        self.mintree = MinTree(capacity)
        self._alpha = alpha
        self._beta = beta
        self._epsilon = 0.00001
        self._max_priority = 0.0

    def _preproc_priority(self, error):
        return (error + self._epsilon) ** self._alpha

    def add(self, obs, action, reward, obs_next, term, priority=None):
        if priority is None:
            priority = self._max_priority
        super(ProportionalReplay, self).add(obs, action, reward, obs_next, term)
        self.sumtree.append(self._preproc_priority(priority))
        self.mintree.append(self._preproc_priority(priority))

    def sample(self):
        idxs = []
        proportion = self.sumtree.sum() / self._batch_size
        for i in range(self._batch_size):
            sum_from = proportion * i
            sum_to = proportion * (i + 1)
            s = random.uniform(sum_from, sum_to)
            idxs.append(self.sumtree.find_sum_idx(s))
        gather = itemgetter(*idxs)
        next_obs_gather = itemgetter(*[i + 1 for i in idxs])
        importances = self._compute_importance(idxs, self._beta)
        return (gather(self._obs),
                gather(self._actions),
                gather(self._rewards),
                next_obs_gather(self._obs),
                gather(self._terms),
                idxs,
                importances)

    def _compute_importance(self, indexes, beta):
        importances = [0.0] * len(indexes)
        if self.mintree.min() == float('inf'):
            return importances
        prob_min = self.mintree.min() / self.sumtree.sum()
        weight_max = (prob_min * self.sumtree.size) ** (-beta)
        for i, idx in enumerate(indexes):
            prob = self.sumtree[idx] / self.sumtree.sum()
            weight = (prob * self.sumtree.size) ** (-beta)
            importances[i] = weight / weight_max
        return importances

    def update(self, indexes, priorities):
        if not isinstance(priorities, np.ndarray):
            priorities = np.asarray(priorities)
        priorities += self._epsilon
        priorities = self._preproc_priority(priorities)
        for idx, prior in zip(indexes, priorities):
            self._max_priority = max(self._max_priority, prior)
            self.sumtree.update(int(idx), prior)
