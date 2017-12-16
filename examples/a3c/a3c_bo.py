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
from reinforceflow.agents import A3CAgent
from reinforceflow.envs.gym_wrapper import AtariWrapper
from reinforceflow.core import EGreedyPolicy
from reinforceflow.nets import A3CConvFactory
reinforceflow.set_random_seed(555)


env_name = 'BreakoutNoFrameskip-v4'
env = AtariWrapper(env_name,
                   action_repeat=4,
                   obs_stack=4,
                   new_width=84,
                   new_height=84,
                   noop_action=[1, 0, 0, 0],
                   start_action=[0, 1, 0, 0])


policies = [EGreedyPolicy(eps_start=1.0, eps_final=final, anneal_steps=4000000)
            for final in [0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.5, 0.5] * 2]

agent = A3CAgent(env, net_factory=A3CConvFactory(), policy=policies, log_every_sec=1200,
                 num_threads=16)
agent.train(log_dir='/tmp/reinforceflow/%s/%s/paper16' % (env_name, agent.name),
            test_episodes=1, test_render=True)
