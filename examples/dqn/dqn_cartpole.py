from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import reinforceflow
except ImportError:
    import os.path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    import reinforceflow
from reinforceflow.agents import DQNAgent
from reinforceflow.envs import GymWrapper
from reinforceflow.core import EGreedyPolicy
from reinforceflow.core import Adam
from reinforceflow.nets import MLPFactory
reinforceflow.set_random_seed(555)

env_name = 'CartPole-v0'
env = GymWrapper(env_name)
policy = EGreedyPolicy(eps_start=1.0, eps_final=0.9, anneal_steps=100000)

agent = DQNAgent(env,
                 net_factory=MLPFactory(),
                 policy=policy,
                 log_every_sec=30,
                 steps=500000,
                 target_freq=1000,
                 optimizer=Adam(0.0001))

agent.train(log_dir='/tmp/reinforceflow/%s/%s/test' % (env_name, agent.name))
