from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import reinforceflow
from reinforceflow.agents.dqn_agent import DQNAgent
from reinforceflow.core import ExperienceReplay
from reinforceflow.nets import dqn
from reinforceflow.envs import EnvFactory
reinforceflow.set_random_seed(321)


steps = 10000000
env = EnvFactory.make('Breakout-v0')
opt = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95, epsilon=0.01)
agent = DQNAgent(env, net_fn=dqn, decay='poly', decay_poly_steps=steps)
agent.train(max_steps=steps,
            log_dir='/tmp/reinforceflow/%s/rms_paper/' % env.spec.id[:-3],
            experience=ExperienceReplay(5000, batch_size=32, min_size=0))
