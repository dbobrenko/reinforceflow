from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class AgentCallback(object):
    def on_iter_start(self, agent, logs):
        pass

    def on_iter_end(self, agent, logs):
        pass

    def on_log(self, agent, logs):
        pass
