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
from reinforceflow.envs.wrapper import Vectorize
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.models import FullyConnected
from reinforceflow.core.optimizer import Adam
from reinforceflow.trainers.async_trainer import AsyncTrainer
reinforceflow.set_random_seed(555)


env_name = 'CartPole-v0'

agent = DeepQ(Vectorize(env_name), model=FullyConnected(), optimizer=Adam(3e-5))

threads = []
for i, eps in enumerate([0.8, 0.4]*2):
    threads.append(DeepQ(Vectorize(env_name),
                   model=FullyConnected(),
                   optimizer=agent.opt,
                   trainable_weights=agent.weights,
                   target_net=agent.target_net,
                   target_weights=agent.target_weights,
                   targetfreq=10000,
                   policy=EGreedyPolicy(1, eps, 100000),
                   name='Thread%s' % i))


trainer = AsyncTrainer(agent,
                       threads,
                       maxsteps=100000,
                       batch_size=20,
                       logdir='/tmp/rf/AsyncDeepQ/%s' % env_name,
                       logfreq=10
                       )
trainer.train()
