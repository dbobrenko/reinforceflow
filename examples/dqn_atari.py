from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import reinforceflow
from reinforceflow.agents.dqn_agent import DQNAgent
from reinforceflow.core import ExperienceReplay
from reinforceflow.nets import dqn
from reinforceflow.envs import EnvFactory
from reinforceflow.core import EGreedyPolicy

reinforceflow.set_random_seed(321)
steps = 10000000
env = EnvFactory.make('Pong-v0')
opt = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95, epsilon=0.01)
agent = DQNAgent(env, net_fn=dqn, decay='poly', decay_poly_steps=steps)
agent.train(max_steps=steps,
            render=True,   # Comment/Remove this line to speed-up training
            log_dir='/tmp/reinforceflow/%s/rms_paper/' % env.spec.id[:-3],
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=1000000),
            experience=ExperienceReplay(size=1000000, min_size=50000, batch_size=32))
