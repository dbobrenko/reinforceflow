from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
from collections import deque


class ExperienceReplay(object):
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def add(self, value):
        self.memory.append(value)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    @property
    def size(self):
        return len(self.memory)
