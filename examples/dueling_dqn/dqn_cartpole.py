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
from reinforceflow.nets import DuelingMLPFactory
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.envs.env_factory import EnvFactory
from reinforceflow.core.replay import ExperienceReplay, ProportionalReplay
reinforceflow.set_random_seed(11)


env_name = 'CartPole-v0'
env = EnvFactory.make(env_name, use_smart_wrap=True)
steps = 8000
agent = DQNAgent(env, net_factory=DuelingMLPFactory(), use_double=False, use_gpu=True)
agent.train(max_steps=steps,
            render=False,
            optimizer='adam',
            learning_rate=0.00005,
            target_freq=500,
            # replay=ExperienceReplay(capacity=steps, batch_size=32, min_size=1024),
            replay=ProportionalReplay(capacity=steps, batch_size=32, min_size=1024),
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=6000),
            log_freq=500,
            ignore_checkpoint=True,
            log_dir='/tmp/reinforceflow/%s/dueling_dqn/uniform' % env_name)
