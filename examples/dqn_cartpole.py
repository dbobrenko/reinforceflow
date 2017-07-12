from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# This try-except block needs only if you haven't installed reinforceflow
try:
    import reinforceflow
except ImportError:
    import os.path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import reinforceflow

from reinforceflow.agents.dqn import DQNAgent
from reinforceflow.nets import mlp
from reinforceflow.envs import EnvFactory
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.core.experience import ExperienceReplay
reinforceflow.set_random_seed(321)


env = EnvFactory.make('CartPole-v0')
steps = 70000
agent = DQNAgent(env, net_fn=mlp)
agent.train(max_steps=steps,
            render=False,
            optimizer='rms',
            learning_rate=0.0001,
            log_dir='/tmp/reinforceflow/%s/rms' % env.spec.id[:-3],
            target_freq=5000,
            experience=ExperienceReplay(size=5000, batch_size=32, min_size=500),
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=30000))
