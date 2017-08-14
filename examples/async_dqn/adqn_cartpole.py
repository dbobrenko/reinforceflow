from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

try:
    import reinforceflow
except ImportError:
    import os.path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import reinforceflow
from reinforceflow.agents.async_dqn import AsyncDQNAgent
from reinforceflow.nets import mlp
from reinforceflow.core.policy import EGreedyPolicy
reinforceflow.set_random_seed(555)

env = 'CartPole-v0'
steps = 220000
agent = AsyncDQNAgent(env, net_fn=mlp, use_gpu=True)
lr = 0.001
agent.train(num_threads=8,
            render=False,
            steps=steps,
            optimizer='adam',
            learning_rate=lr,
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=steps / 10),
            target_freq=40000,
            gamma=0.99,
            batch_size=5,
            log_freq=50000,
            log_dir='/tmp/reinforceflow/%s/async_dqn/adam/' % env)
