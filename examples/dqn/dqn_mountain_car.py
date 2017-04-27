import gym
import tensorflow as tf
import reinforceflow
from reinforceflow.agents.dqn_agent import DqnAgent
from reinforceflow.nets import mlp
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.core.experience import ExperienceReplay
reinforceflow.set_random_seed(322)


# env = gym.make('SpaceInvaders-v0')
# env = gym.make('CartPole-v0')
env = gym.make('MountainCar-v0')
opt = tf.train.RMSPropOptimizer(0.1, momentum=0.95, epsilon=0.01)
# opt = tf.train.AdamOptimizer(0.1)
agent = DqnAgent(env, log_dir='/tmp/rf/MountainCar/dqn-1e6/', net_fn=mlp, opt=opt)
agent.fit(max_steps=1e6,
          render=False,
          target_freq=10000,
          experience_min=5000,
          experience=ExperienceReplay(size=100000),
          policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=100000))
