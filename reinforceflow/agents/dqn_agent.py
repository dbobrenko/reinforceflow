from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range
import numpy as np
import tensorflow as tf
from reinforceflow.agents import DiscreteAgent
from reinforceflow.core import ExperienceReplay
from reinforceflow.nets import dqn
from reinforceflow.core import EGreedyPolicy, GreedyPolicy
from reinforceflow import misc
from reinforceflow import logger
# TODO: Test & write unittests
# TODO: Add comments & documentation
# TODO: Log more info to tensorboard
# TODO: Make Environment Factory
# TODO: Make preprocessing function (in graph)


class DqnAgent(DiscreteAgent):
    def __init__(self,
                 env,
                 sess=None,
                 net_fn=dqn,
                 epsilon=0.99,
                 gamma=0.99,
                 gradient_clip=40.0,
                 log_dir='/tmp/rf/',
                 opt=tf.train.RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01),
                 learning_rate=None,
                 decay='poly',
                 decay_poly_end_lr=0.0001,
                 decay_poly_power=1.0,
                 decay_poly_steps=1e7,
                 decay_rate=0.96):
        """Constructs Deep Q-Network agent;
        Based on paper "Human-level control through deep reinforcement learning", Mnih et al., 2015.
        Current agent solves environments with discrete _action spaces. Initially designed to work with raw pixel inputs.

        Args:
            env (reinforceflow.EnvWrapper): Environment wrapper
            sess: TensorFlow Session
            net_fn: Function, that takes `input_shape` and `output_size` arguments,
                    and returns (input Tensor, output Tensor, all end point Tensors)
            epsilon (float): The probability for epsilon-greedy exploration, expected to be in range [0; 1]
            gamma (float): Discount factor
            gradient_clip (float): Norm gradient clipping, to disable, pass 0 or None
            log_dir (str): path to directory, where all agent's outputs will be saved (session, summary, logs, etc)
            opt: An optimizer instance, optimizer name, or optimizer class
            learning_rate (float or Tensor): Should be provided, if `opt` is optimizer class or name
            decay: Learning rate decay. Should be provided decay function, or decay function name.
                   Available decays: 'polynomial', 'exponential'. To disable decay, pass None.
            decay_poly_end_lr (float or Tensor): The minimal end learning rate.
                                                 Should be provided, if polynomial decay was chosen.
            decay_poly_power (float or Tensor): The power of the polynomial.
                                                E.g. `power=1.0` means linear learning rate decay.
                                                Should be provided, if polynomial decay was chosen.
            decay_poly_steps (int or Tensor): The number of steps over which learning rate anneals
                                              down to `decay_poly_end_lr`.
                                              Should be provided, if polynomial decay was chosen.
            decay_rate (float): The decay rate. Should be provided, if exponential decay was chosen.

        Attributes:
            env: Agent's running environment
            sess: TensorFlow Session
            global_step (Tensor): TensorFlow global step
            opt: TensorFlow optimizer
            lr: TensorFlow optimizer's learning rate
        """
        super(DqnAgent, self).__init__(env=env, epsilon=epsilon, gamma=gamma)
        self.log_dir = log_dir
        self.sess = tf.Session() if sess is None else sess
        with tf.variable_scope('network'):
            self._action = tf.placeholder('int32', [None], name='_action')
            self._reward = tf.placeholder('float32', [None], name='_reward')
            self._obs, self._q, _ = net_fn(input_shape=[None] + self.env.obs_shape, output_size=self.env.action_shape)

        with tf.variable_scope('target_network'):
            self._target_obs, self._target_q, _ = net_fn(input_shape=[None] + self.env.obs_shape,
                                                         output_size=self.env.action_shape)

        with tf.variable_scope('_target_update'):
            target_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_network')
            w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'network')
            self._target_update = [target_w[i].assign(w[i]) for i in range(len(target_w))]

        with tf.variable_scope('optimizer'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.opt, self.lr = misc.create_optimizer(opt, learning_rate, decay=decay,
                                                      decay_poly_steps=decay_poly_steps,
                                                      decay_poly_end_lr=decay_poly_end_lr,
                                                      decay_poly_power=decay_poly_power,
                                                      decay_rate=decay_rate)
            action_one_hot = tf.one_hot(self._action, self.env.action_shape, 1.0, 0.0)
            # Predict expected future _reward for performed _action
            q_value = tf.reduce_sum(tf.multiply(self._q, action_one_hot), axis=1)
            self._loss = tf.reduce_mean(tf.square(self._reward - q_value))
            grads = tf.gradients(self._loss, w)
            if gradient_clip:
                grads, _ = tf.clip_by_global_norm(grads, gradient_clip)
            grads_vars = list(zip(grads, w))
            self._train_op = self.opt.apply_gradients(grads_vars, global_step=self.global_step)

        self._log_vars = {'metrics/trainR': 0, 'metrics/trainQ': 0, 'metrics/testR': 0, 'metrics/testQ': 0, 'lr': 0}
        self.summary = misc.AgentSummary(self.sess, log_dir, *tf.trainable_variables(), self.lr,
                                         scalar_tags=list(self._log_vars.keys()))
        # TODO print final config

    def predict(self, obs):
        return self.sess.run(self._q, {self._obs: obs})

    def predict_target(self, obs):
        return self.sess.run(self._target_q, {self._target_obs: obs})

    def update_target(self):
        self.sess.run(self._target_update)

    def train_on_batch(self, obs, actions, rewards):
        """Trains agent on given batch.

        Args:
            obs (nd.array): input observations with shape=[batch, height, width, channels]
            actions: list of actions
            rewards: list with rewards for each action
        """
        self.sess.run(self._train_op, feed_dict={
            self._obs: obs,
            self._action: actions,
            self._reward: rewards,
        })

    def play(self, episodes, policy=GreedyPolicy(), max_steps=1e5, render=False):
        """Tests agent's performance with specified policy on given number of games"""
        ep_rewards = []
        ep_q = []
        for _ in range(episodes):
            ep_reward = 0
            obs = self.env.reset()
            for step in range(max_steps):
                if render:
                    self.env.render()
                reward_per_action = self.predict(obs)
                action = policy.select_action(reward_per_action, env=self.env)
                obs, r, terminal, info = self.env.step(action)
                ep_q.append(np.max(reward_per_action))
                ep_reward += r
                if terminal or step > max_steps:
                    break
            ep_rewards.append(ep_reward)
        return ep_rewards, ep_q

    def fit(self,
            render,
            max_steps=1e7,
            batch_size=32,
            target_freq=10000,
            gamma=0.99,
            experience_min=50000,
            experience=ExperienceReplay(size=1000000),
            policy=EGreedyPolicy(eps_start=1.0, eps_final=0.1, anneal_steps=1000000),
            log_freq=2000):
        total_reward = 0
        ep_num = 0
        ep_reward = 0
        q_total = 0
        q_num = 0
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            obs = self.env.reset()
            for step in range(max_steps):
                if render:
                    self.env.render()
                reward_per_action = self.predict(obs)
                action = policy.select_action(reward_per_action, self.env, step)
                obs_next, reward, term, info = self.env.step(action)
                ep_reward += reward
                reward = np.clip(reward, -1, 1)
                experience.add({'obs': obs, 'action': action, 'reward': reward, 'obs_next': obs_next, 'term': term})
                obs = obs_next
                if term:
                    ep_num += 1
                    total_reward += ep_reward
                    ep_reward = 0
                    obs = self.env.reset()
                if experience.size > experience_min:
                    batch = experience.sample(batch_size)
                    tr_obs = []
                    tr_action = []
                    tr_reward = []
                    for transition in batch:
                        tr_obs.append(transition['obs'])
                        tr_action.append(transition['action'])
                        td_target = transition['reward']
                        if not transition['term']:
                            q = np.max(self.predict_target(transition['obs_next']).flatten())
                            td_target += gamma * q
                            q_total += q
                            q_num += 1
                        tr_reward.append(td_target)
                    self.train_on_batch(np.vstack(tr_obs), tr_action, tr_reward)

                    if step % target_freq == 0:
                        self.update_target()

                    # Eval & log
                    if step % log_freq == 0:
                        tf.train.Saver().save(self.sess, self.log_dir, step)
                        # Behaviour policy evaluation
                        self._log_vars['metrics/trainR'] = total_reward / ep_num
                        self._log_vars['metrics/trainQ'] = q_total / q_num
                        q_total = 0
                        q_num = 0
                        ep_num = 0
                        total_reward = 0
                        logger.info("Behaviour policy: Avg.Ep.R: %.4f. Avg.Ep.Q: %.2f. Step: %d" %
                                    (self._log_vars['metrics/trainR'], self._log_vars['metrics/trainQ'], step))

                        # Greedy policy evaluation
                        test_r, test_q = self.play(episodes=5)
                        self._log_vars['metrics/testR'] = np.mean(test_r)
                        self._log_vars['metrics/testQ'] = np.mean(test_q)
                        logger.info("Greedy policy: Avg.Ep.R: %.4f. Avg.Ep.Q: %.2f. Step: %d" %
                                    (self._log_vars['metrics/testR'], self._log_vars['metrics/testQ'], step))
                        self.summary.write_summary(sess=self.sess, step=step, summary_dict=self._log_vars)
