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
from reinforceflow.core import ExperienceReplay
from reinforceflow.nets import DQNFactory
from reinforceflow.core import EGreedyPolicy
from reinforceflow.envs.gym_wrapper import GymPixelWrapper
reinforceflow.set_random_seed(555)

steps = 50000000
env_name = 'PongNoFrameskip-v4'
env = GymPixelWrapper(env_name,
                      action_repeat=4,
                      obs_stack=4,
                      resize_width=84,
                      resize_height=84)
optimizer_args = {'momentum': 0.95}
replay_size = 20000


agent = DQNAgent(env, net_factory=DQNFactory(), use_double=True, use_gpu=True)
agent.train(max_steps=steps,
            render=False,
            optimizer='rms',
            learning_rate=0.00025,
            optimizer_args=optimizer_args,
            update_freq=4,
            target_freq=10000,
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=1000000),
            replay=ExperienceReplay(capacity=replay_size, min_size=replay_size, batch_size=32),
            log_every_sec=2000,
            log_dir='/tmp/reinforceflow/%s/double_dqn/rms_paper/' % env_name)
