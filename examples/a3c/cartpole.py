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
from reinforceflow.agents import ActorCritic
from reinforceflow.envs.wrapper import Vectorize
from reinforceflow.models import ActorCriticFC
from reinforceflow.core.optimizer import RMSProp
from reinforceflow.trainers.async_trainer import AsyncTrainer

reinforceflow.set_random_seed(555)


env_name = 'CartPole-v0'

agent = ActorCritic(Vectorize(env_name),
                    model=ActorCriticFC(),
                    optimizer=RMSProp(7e-4, decay=0.99, epsilon=0.1))
threads = []
for i in range(2):
    threads.append(ActorCritic(Vectorize(env_name),
                               model=ActorCriticFC(),
                               optimizer=agent.opt,
                               trainable_weights=agent.weights,
                               name='thread%s' % i))


trainer = AsyncTrainer(agent,
                       threads,
                       maxsteps=500000,
                       batch_size=20,
                       logdir='/tmp/rf/A3C/%s' % env_name,
                       logfreq=10
                       )
trainer.train()
