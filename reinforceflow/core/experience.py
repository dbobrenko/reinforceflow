from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
from collections import deque


class ExperienceReplay(object):
    def __init__(self, size, min_size, batch_size):
        self.memory = deque(maxlen=size)
        self.min_size = min_size
        self.batch_size = batch_size

    def add(self, value):
        self.memory.append(value)

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    @property
    def size(self):
        return len(self.memory)

    @property
    def is_ready(self):
        return self.size >= self.min_size + self.batch_size

    def __len__(self):
        return len(self.memory)

