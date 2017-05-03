from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import tensorflow as tf
import reinforceflow
from reinforceflow.agents.dqn_agent import DQNAgent
from reinforceflow.nets import mlp
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.core.experience import ExperienceReplay
reinforceflow.set_random_seed(321)


env = gym.make('CartPole-v0')
opt = tf.train.RMSPropOptimizer(learning_rate=0.0001)
steps = 70000
agent = DQNAgent(env, net_fn=mlp, opt=opt, decay=None, decay_poly_steps=steps)
agent.train(max_steps=steps,
            log_dir='/tmp/reinforceflow/CartPole/test4/',
            render=False,
            target_freq=5000,
            experience_min=500,
            # checkpoint='/tmp/rf/MountainCar/dqn-sgd0.2-del2/',
            experience=ExperienceReplay(size=5000),
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=100000))
