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
from reinforceflow.envs.env_factory import EnvFactory
from reinforceflow.nets import dqn
reinforceflow.set_random_seed(4)


env = EnvFactory.make('SpaceInvaders-v0', use_smart_wrap=True)
steps = 80000000
agent = AsyncDQNAgent(env, net_fn=dqn, use_gpu=True)
agent.train(num_threads=8,
            render=False,
            steps=steps,
            optimizer='adam',
            learning_rate=0.0003,
            epsilon_pool=4*[0.1] + 3*[0.01] + 3*[0.5],
            epsilon_steps=4000000,
            target_freq=40000,
            gamma=0.99,
            batch_size=5,
            log_freq=100000,
            log_dir='/tmp/reinforceflow/async_dqn/SpaceInvaders-v0/adam_8thr/')
