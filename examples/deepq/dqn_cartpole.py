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
from reinforceflow.agents import DeepQ
from reinforceflow.envs import Vectorize
from reinforceflow.core import EGreedyPolicy
from reinforceflow.core import Adam
from reinforceflow.models import FullyConnected
reinforceflow.set_random_seed(555)

env_name = 'CartPole-v0'
env = Vectorize(env_name)
policy = EGreedyPolicy(eps_start=1.0, eps_final=0.2, anneal_steps=300000)

agent = DeepQ(env, model=FullyConnected(), optimizer=Adam(0.0001))

agent.train(policy=policy,
            log_freq=10,
            maxsteps=300000,
            target_freq=1000,
            log_dir='/tmp/rf/DeepQ/%s' % env_name)
