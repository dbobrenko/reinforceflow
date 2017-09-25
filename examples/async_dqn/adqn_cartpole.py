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
from reinforceflow.agents.async_dqn import AsyncDQNAgent
from reinforceflow.envs.gym_wrapper import GymWrapper
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.nets import MLPFactory
reinforceflow.set_random_seed(555)

env_name = 'CartPole-v0'
env = GymWrapper(env_name)
steps = 60000
agent = AsyncDQNAgent(env, net_factory=MLPFactory(layer_sizes=(256, 256)), use_gpu=True)
agent.train(num_threads=8,
            render=False,
            steps=steps,
            optimizer='adam',
            learning_rate=0.0001,
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.5, anneal_steps=steps / 4),
            target_freq=5000,
            gamma=0.99,
            batch_size=5,
            log_every_sec=30,
            ignore_checkpoint=True,  # Always starts training from scratch
            test_render=True,  # Renders evaluation tests.
            log_dir='/tmp/reinforceflow/%s/async_dqn/adam/' % env_name)

agent.test(10, render=True, max_fps=30)
