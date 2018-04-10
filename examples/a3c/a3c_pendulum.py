"""A3C with continious action space example."""
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
from reinforceflow.agents.async.a3c import A3C
from reinforceflow.envs.wrapper import Vectorize
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.models import A3CMLPFactory
reinforceflow.set_random_seed(555)

env_name = 'Pendulum-v0'
env = Vectorize(env_name)
steps = 200000
agent = A3C(env, model=A3CMLPFactory(layer_sizes=(256, 256)),
            use_gpu=True)
agent.train(num_threads=4,
            render=False,
            steps=steps,
            optimizer='adam',
            learning_rate=0.0005,
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=0.8 * steps),
            target_freq=5000,
            gamma=0.99,
            batch_size=20,
            log_freq=20,
            ignore_checkpoint=True,
            log_dir='/tmp/reinforceflow/%s/a3c/adam/' % env_name)

agent.test(False, 10, render=True, max_fps=30)
