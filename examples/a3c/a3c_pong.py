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
from reinforceflow.envs.gym_wrapper import GymPixelWrapper
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.nets import A3CConvFactory
reinforceflow.set_random_seed(555)


env_name = 'PongNoFrameskip-v4'
env = GymPixelWrapper(env_name,
                      action_repeat=4,
                      obs_stack=4,
                      resize_width=84,
                      resize_height=84,
                      to_gray=True)
steps = 80000000
agent = A3CAgent(env, net_factory=A3CConvFactory(), use_gpu=True)
policies = [EGreedyPolicy(eps_start=1.0, eps_final=final, anneal_steps=4000000)
            for final in [0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.5, 0.5]]
agent.train(num_threads=8,
            render=False,
            steps=steps,
            optimizer='adam',
            learning_rate=0.0003,
            policy=policies,
            target_freq=40000,
            gamma=0.99,
            batch_size=5,
            log_every_sec=1800,
            test_render=True,
            log_dir='/tmp/reinforceflow/%s/a3c/adam_3e4/' % env_name)
