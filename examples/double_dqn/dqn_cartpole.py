from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

try:
    import reinforceflow
except ImportError:
    import os.path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    import reinforceflow
from reinforceflow.agents.dqn import DQNAgent
from reinforceflow.nets import MLPFactory
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.core.replay import ExperienceReplay, ProportionalReplay
reinforceflow.set_random_seed(555)


env = 'CartPole-v0'
steps = 30000
agent = DQNAgent(env, net_factory=MLPFactory(), use_double=True, use_gpu=True)
agent.train(max_steps=steps,
            render=False,
            optimizer='adam',
            learning_rate=0.0001,
            target_freq=500,
            replay=ExperienceReplay(capacity=20000, batch_size=32, min_size=1024),
            # replay=ProportionalReplay(capacity=20000, batch_size=32, min_size=1024),
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=20000),
            log_freq=500,
            log_dir='/tmp/reinforceflow/%s/double_dqn/uniform' % env)

