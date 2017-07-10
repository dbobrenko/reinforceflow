from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import reinforceflow
from reinforceflow.agents.async_dqn import AsyncDQNAgent
from reinforceflow.nets import mlp
reinforceflow.set_random_seed(321)


env = 'CartPole-v0'
steps = 500000
agent = AsyncDQNAgent(env, net_fn=mlp)
lr = 0.000001
agent.train(num_threads=8,
            render=False,
            steps=steps,
            optimizer='adam',
            learning_rate=lr,
            epsilon_steps=steps / 10,
            target_freq=10000,
            batch_size=32,
            log_freq=5000,
            log_dir='/tmp/reinforceflow/%s/adam_%s/' % (env, lr))
