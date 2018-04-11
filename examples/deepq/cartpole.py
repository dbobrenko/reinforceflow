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
from reinforceflow.core import EGreedyPolicy, ProportionalReplay
from reinforceflow.core import Adam
from reinforceflow.models import FullyConnected
from reinforceflow.trainers.replay_trainer import ReplayTrainer
reinforceflow.set_random_seed(555)

env_name = 'CartPole-v0'
env = Vectorize(env_name)
policy = EGreedyPolicy(eps_start=1.0, eps_final=0.2, anneal_steps=300000)

agent = DeepQ(env,
              model=FullyConnected(),
              optimizer=Adam(0.0001),
              targetfreq=10000,
              policy=EGreedyPolicy(1, 0.4, 300000),
              trajectory_batch=False)

trainer = ReplayTrainer(env=env,
                        agent=agent,
                        maxsteps=300000,
                        replay=ProportionalReplay(30000, 32, 32),
                        logdir='/tmp/rf/DeepQ/%s' % env_name,
                        logfreq=10)
trainer.train()
