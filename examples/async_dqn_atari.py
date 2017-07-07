from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import reinforceflow
from reinforceflow.agents.async_dqn import AsyncDQNTrainer
from reinforceflow.nets import dqn
reinforceflow.set_random_seed(321)

env = 'SpaceInvaders-v0'
steps = 80000000
agent = AsyncDQNTrainer(env, net_fn=dqn)
agent.train(num_threads=8,
            render=False,
            steps=steps,
            optimizer='rms',
            learning_rate=0.0001,
            epsilon_pool=4*[0.1] + 3*[0.01] + 3*[0.5],
            epsilon_steps=4000000,
            target_freq=10000,
            gamma=0.99,
            batch_size=32,
            log_freq=5000,
            log_dir='/tmp/reinforceflow/%s/rms/' % env)
