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
from reinforceflow.agents import AsyncDQNAgent
from reinforceflow.envs.gym_wrapper import GymWrapper
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.nets import A3CMLPFactory
reinforceflow.set_random_seed(555)
from reinforceflow.core.optimizer import Adam

env_name = 'CartPole-v0'
env = GymWrapper(env_name)
policies = EGreedyPolicy(eps_start=1.0, eps_final=0.9, anneal_steps=100000)

agent = AsyncDQNAgent(env,
                      net_factory=A3CMLPFactory(),
                      policy=policies,
                      log_every_sec=20,
                      steps=100000,
                      optimizer=Adam(3e-5),
                      batch_size=20)

agent.train(log_dir='/tmp/reinforceflow/%s/%s/test' % (env_name, agent.name))
agent.test(10, render=True, max_fps=30)
