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
from reinforceflow.agents import AsyncDeepQ
from reinforceflow.envs.wrapper import Vectorize
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.models import FullyConnected
from reinforceflow.core.optimizer import Adam
reinforceflow.set_random_seed(555)


env_name = 'CartPole-v0'
env = Vectorize(env_name)
policies = EGreedyPolicy(eps_start=1.0, eps_final=0.9, anneal_steps=100000)

agent = AsyncDeepQ(env, model=FullyConnected(), optimizer=Adam(3e-5))

agent.train(log_freq=30,
            test_env=Vectorize(env_name),
            render=False,
            policy=policies,
            maxsteps=100000,
            batch_size=20,
            log_dir='/tmp/rf/AsyncDeepQ/%s' % env_name)
