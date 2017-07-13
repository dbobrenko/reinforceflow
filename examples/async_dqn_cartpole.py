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
from reinforceflow.agents.async_dqn import AsyncDQNAgent
from reinforceflow.nets import mlp
reinforceflow.set_random_seed(321)

env = 'CartPole-v0'
steps = 300000
agent = AsyncDQNAgent(env, net_fn=mlp, use_gpu=False)
lr = 0.01
agent.train(num_threads=4,
            render=False,
            steps=steps,
            optimizer='adam',
            learning_rate=lr,
            epsilon_steps=steps / 10,
            target_freq=40000,
            gamma=0.99,
            batch_size=32,
            log_freq=20000,
            log_dir='/tmp/reinforceflow/async_dqn/%s/adam_%s/' % (env, lr))
