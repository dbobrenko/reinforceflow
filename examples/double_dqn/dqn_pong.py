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
from reinforceflow.nets import DuelingMLPFactory
from reinforceflow.core import EGreedyPolicy
from reinforceflow.envs import EnvFactory
reinforceflow.set_random_seed(321)

steps = 10000000
env_name = 'Breakout-v0'
env = EnvFactory.make(env_name, use_smart_wrap=True)
optimizer_args = {'momentum': 0.95, 'epsilon': 0.01}
decay_args = {'power': 1.0, 'decay_steps': steps}
# DQN authors used replay buffer of size 1000000 (1 million of frames).
# Since the entire buffer lives in RAM, it will require 100+ GB of memory.
# 20000 was chosen to save some memory, and it seems to be converging.
# However, it can be increased.
replay_size = 20000


agent = DQNAgent(env, net_factory=DuelingMLPFactory(), use_double=True, use_gpu=True)
agent.train(max_steps=steps,
            render=False,
            optimizer='rms',
            learning_rate=0.00025,
            optimizer_args=optimizer_args,
            decay='poly',
            decay_args=decay_args,
            log_dir='/tmp/reinforceflow/%s/double_dqn/rms_paper/' % env_name,
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=1000000),
            replay=ExperienceReplay(capacity=replay_size, min_size=20000, batch_size=32),
            log_freq=2000)
