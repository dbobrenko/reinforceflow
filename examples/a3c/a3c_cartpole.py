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
from reinforceflow.agents.a3c import A3CAgent
from reinforceflow.envs.env_factory import EnvFactory
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.nets import A3CMLPFactory
reinforceflow.set_random_seed(555)

env_name = 'CartPole-v0'
env = EnvFactory.make(env_name, use_smart_wrap=True)
steps = 80000
agent = A3CAgent(env, net_factory=A3CMLPFactory(layer_sizes=(256, 256)), use_gpu=True)
agent.train(num_threads=8,
            render=False,
            steps=steps,
            optimizer='adam',
            learning_rate=0.005,
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=0.8 * steps),
            target_freq=5000,
            gamma=0.99,
            batch_size=20,
            log_freq=5000,
            ignore_checkpoint=True,
            log_dir='/tmp/reinforceflow/%s/a3c/adam/' % env_name)
