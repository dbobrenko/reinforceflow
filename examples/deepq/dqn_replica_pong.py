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
from reinforceflow.agents.deepq import DeepQ
from reinforceflow.models import DeepQModel
from reinforceflow.envs.atari import AtariWrapper
from reinforceflow.core import ExperienceReplay, EGreedyPolicy

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
                        to_gray=True)

agent = DeepQ(env=env,
              model=DeepQModel(),
              use_double=False)

agent.train(maxsteps=50000000,
            log_freq=60,
            lr_schedule='linear',
            replay=ExperienceReplay(30000, 32, 32),
            render=False,
            test_env=test_env,
            test_render=False,
            test_episodes=1,
            log_dir='/tmp/rf/DeepQ/%s' % env_name)
