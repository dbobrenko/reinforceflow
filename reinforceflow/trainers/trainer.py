from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class BaseTrainer(object):
    def train(self, **kwargs):
        pass

    def save(self):
        pass

    def load(self):
        pass
