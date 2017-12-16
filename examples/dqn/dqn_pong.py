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
from reinforceflow.agents.dqn import DQNAgent
from reinforceflow.nets import DQNFactory
from reinforceflow.envs.gym_wrapper import AtariWrapper
from reinforceflow.core import ExperienceReplay, EGreedyPolicy, Adam

reinforceflow.set_random_seed(555)

env_name = 'PongNoFrameskip-v0'
env = AtariWrapper(env_name,
                   action_repeat=4,
                   obs_stack=4,
                   new_width=84,
                   new_height=84,
                   to_gray=True)


agent = DQNAgent(env,
                 net_factory=DQNFactory(),
                 use_double=False,
                 target_freq=1000,
                 steps=int(50e6),
                 optimizer=Adam(1e-4),
                 policy=EGreedyPolicy(1.0, 0.02, 100000),
                 exp=ExperienceReplay(100000, 32, 10000),
                 log_every_sec=1200)

agent.train(log_dir='/tmp/reinforceflow/%s/%s/adam1e-4' % (env_name, agent.name))
