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
from reinforceflow.agents.async.asyncdeepq import AsyncDeepQ
from reinforceflow.envs.atari import AtariWrapper
from reinforceflow.core import EGreedyPolicy
from reinforceflow.models import DeepQModel
reinforceflow.set_random_seed(555)

env_name = 'PongNoFrameskip-v4'
env = AtariWrapper(env_name,
                   action_repeat=4,
                   obs_stack=4,
                   new_width=84,
                   new_height=84,
                   noop_action=[1, 0, 0, 0, 0, 0],
                   fire_action=[0, 1, 0, 0, 0, 0])


policies = [EGreedyPolicy(eps_start=1.0, eps_final=final, anneal_steps=4000000)
            for final in [0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.5, 0.5] * 2]

agent = AsyncDeepQ(env, model=DeepQModel(nature_arch=False, dueling=False), num_threads=16)

agent.train(maxsteps=80000000,
            policy=policies,
            log_freq=120,
            test_episodes=1,
            test_render=False,
            log_dir='/tmp/rf/AsyncDeepQ/%s' % env_name)
