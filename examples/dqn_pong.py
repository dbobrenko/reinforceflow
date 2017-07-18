from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# This try-except block needs only if reinforceflow isn't installed
try:
    import reinforceflow
except ImportError:
    import os.path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import reinforceflow
from reinforceflow.agents.dqn import DQNAgent
from reinforceflow.core import ExperienceReplay
from reinforceflow.nets import dqn
from reinforceflow.core import EGreedyPolicy
reinforceflow.set_random_seed(321)

steps = 10000000
env = EnvFactory.make('Breakout-v0', use_smart_wrap=True)
optimizer_args = {'momentum': 0.95, 'epsilon': 0.01}
decay_args = {'power': 1.0, 'decay_steps': steps}
# Authors of the DQN used replay buffer size of 1000000 (1 million of frames).
# Since the entire buffer lives in RAM, it will require 100+ GB of memory.
# That's why 20000 was chosen by default (~4 GB RAM), and it seems to be converging.
replay_size = 20000

agent = DQNAgent(env, net_fn=dqn, use_double=True, use_gpu=True)
agent.train(max_steps=steps,
            render=False,
            optimizer='rms',
            learning_rate=0.00025,
            optimizer_args=optimizer_args,
            decay='poly',
            decay_args=decay_args,
            log_dir='/tmp/reinforceflow/double_dqn/%s/rms_paper/' % env,
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=1000000),
            experience=ExperienceReplay(size=replay_size, min_size=50000, batch_size=32))
