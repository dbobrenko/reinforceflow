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
from reinforceflow.agents import A3C
from reinforceflow.envs.atari import AtariWrapper
from reinforceflow.core import EGreedyPolicy
from reinforceflow.models import ActorCriticConv
reinforceflow.set_random_seed(555)


env_name = 'PongNoFrameskip-v4'
env = AtariWrapper(env_name,
                   action_repeat=4,
                   obs_stack=4,
                   new_width=84,
                   new_height=84,
                   to_gray=True,
                   noop_action=[1, 0, 0, 0, 0, 0],
                   fire_action=[0, 1, 0, 0, 0, 0],
                   clip_rewards=True)


test_env = AtariWrapper(env_name,
                        action_repeat=4,
                        obs_stack=4,
                        new_width=84,
                        new_height=84,
                        noop_action=None,
                        fire_action=None,
                        clip_rewards=False)

policies = [EGreedyPolicy(eps_start=1.0, eps_final=final, anneal_steps=4000000)
            for final in [0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.5, 0.5] * 2]

agent = A3C(env=env,
            model=ActorCriticConv(),
            num_threads=16)

agent.train(policy=policies,
            render=False,
            maxsteps=80000000,
            lr_schedule='linear',
            log_freq=240,
            test_episodes=1,
            test_render=False,
            log_dir='/tmp/rf/A3C/%s' % env_name)
