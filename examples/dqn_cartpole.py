from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# This try-except block needs only if reinforceflow isn't installed
try:
    import reinforceflow
except ImportError:
    import os.path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import reinforceflow
from reinforceflow.agents.dqn import DQNAgent
from reinforceflow.nets import mlp
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.core.replay import ExperienceReplay, ProportionalReplay
reinforceflow.set_random_seed(11)


env = 'CartPole-v0'
steps = 40000
agent = DQNAgent(env, net_fn=mlp, use_double=True, use_gpu=True)
agent.train(max_steps=steps,
            render=False,
            optimizer='rms',
            learning_rate=0.0001,
            log_dir='/tmp/reinforceflow/double_dqn/%s/rms' % env,
            target_freq=2000,
            experience=ExperienceReplay(capacity=20000, batch_size=32, min_size=1024),
            # experience=ProportionalReplay(capacity=20000, batch_size=32, min_size=1024),
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=20000))
