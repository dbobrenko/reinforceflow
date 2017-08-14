from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class SumTree(object):
    """This code partially based on the sum tree realization from:
       https://github.com/jaara/AI-blog/blob/master/SumTree.py"""
    def __init__(self, capacity, default_priority=0):
        self._size = 0
        self._capacity = capacity
        self._tree = [default_priority] * (2*capacity - 1)
        self._current_idx = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self._tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self._tree):
            return idx
        if s <= self._tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self._tree[left])

    def update(self, idx, priority):
        self._update(self._capacity + idx - 1, priority)

    def _update(self, idx, priority):
        diff = priority - self._tree[idx]
        self._tree[idx] = priority
        self._propagate(idx, diff)

    def sum(self):
        return self._tree[0]

    def append(self, priority):
        idx = self._current_idx + self._capacity - 1
        self._update(idx, priority)
        self._current_idx += 1
        self._size = min(self._size + 1, self._capacity)
        if self._current_idx >= self._capacity:
            self._current_idx = 0

    def find_sum_idx(self, s):
        if s > self.sum():
            s = self.sum()
        idx = self._retrieve(0, s) - self._capacity + 1
        if idx >= self._size:
            idx = self._size - 1
        return idx

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._tree[self._capacity + idx - 1]

    @property
    def size(self):
        return self._size

    @property
    def capacity(self):
        return self._capacity


class MinTree(object):
    def __init__(self, capacity, default_priority=float('inf')):
        self._size = 0
        self._capacity = capacity
        self._tree = [default_priority] * (2*capacity - 1)
        self._current_idx = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        # self._tree[parent] += change
        self._tree[idx] = change
        left = 2 * parent + 1
        right = left + 1
        change = min(self._tree[left], self._tree[right])
        self._tree[parent] = change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self._tree):
            return idx
        if s <= self._tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self._tree[left])

    def update(self, idx, priority):
        self._update(self._capacity + idx - 1, priority)

    def _update(self, idx, priority):
        # diff = priority - self._tree[idx]
        self._tree[idx] = priority
        self._propagate(idx, priority)

    # def sum(self):
    #     return self._tree[0]

    def min(self):
        return self._tree[0]

    def append(self, priority):
        idx = self._current_idx + self._capacity - 1
        self._update(idx, priority)
        self._current_idx += 1
        self._size = min(self._size + 1, self._capacity)
        if self._current_idx >= self._capacity:
            self._current_idx = 0

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._tree[self._capacity + idx - 1]

    @property
    def size(self):
        return self._size

    @property
    def capacity(self):
        return self._capacity
