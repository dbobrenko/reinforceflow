from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from reinforceflow.agents.async.base_async import AsyncAgent, AsyncThreadAgent
from reinforceflow.core import AgentCallback
from reinforceflow.core.agent import BaseDQNAgent
from reinforceflow.core.optimizer import RMSProp
from reinforceflow.utils import discount_rewards
from reinforceflow.utils.tensor_utils import add_observation_summary


class _AsyncDQNCallback(AgentCallback):
    def __init__(self):
        self.last_target_update = 0

    def on_iter_end(self, agent, logs):
        if logs['obs_counter'] - self.last_target_update >= agent.target_freq:
            self.last_target_update = logs['obs_counter']
            agent.target_update()


class AsyncDQNAgent(AsyncAgent, BaseDQNAgent):
    """Constructs Asynchronous N-step Q-Learning agent, based on paper:
    Asynchronous Methods for Deep Reinforcement Learning, Mnih et al., 2016
    (https://arxiv.org/abs/1602.01783v2).

    See `AsyncAgent`, `BaseDQNAgent`.
    Args:
        restore_from (str): Path to the pre-trained model.
        target_freq (int): [Training-only] Target network update frequency (in observations).
    """

    def __init__(self, env, net_factory, restore_from=None, device='/gpu:0', steps=int(80e6),
                 optimizer=None, policy=None, num_threads=8, batch_size=5, gamma=0.99,
                 target_freq=40000, saver_keep=3, log_every_sec=300, name='AsyncDQNAgent'):
        if optimizer is None:
            optimizer = RMSProp(7e-4, decay=0.99, epsilon=0.1, lr_decay='linear')
        super(AsyncDQNAgent, self).__init__(env=env,
                                            net_factory=net_factory,
                                            thread_agent=_ThreadDQNAgent,
                                            device=device,
                                            steps=steps,
                                            optimizer=optimizer,
                                            policy=policy,
                                            num_threads=num_threads,
                                            batch_size=batch_size,
                                            gamma=gamma,
                                            saver_keep=saver_keep,
                                            log_every_sec=log_every_sec,
                                            name=name)
        self.target_freq = target_freq
        self.sess.run(tf.global_variables_initializer())
        if restore_from is not None and tf.train.latest_checkpoint(restore_from) is not None:
            self.load_weights(restore_from)

    def train(self, log_dir, render=False, test_render=False, test_episodes=3, callbacks=set()):
        """Starts training. See `AsyncAgent.train`."""
        callbacks = set(callbacks)
        callbacks.add(_AsyncDQNCallback())
        super(AsyncDQNAgent, self).train(log_dir, render, test_render, test_episodes, callbacks)


class _ThreadDQNAgent(AsyncThreadAgent, BaseDQNAgent):
    def __init__(self, env, net_factory, global_agent, policy, gamma, batch_size,
                 device, log_every_sec, name=''):
        super(_ThreadDQNAgent, self).__init__(env=env,
                                              net_factory=net_factory,
                                              global_agent=global_agent,
                                              policy=policy,
                                              gamma=gamma,
                                              batch_size=batch_size,
                                              device=device,
                                              log_every_sec=log_every_sec,
                                              name=name)
        with tf.device(self.device), tf.variable_scope(self._scope + 'optimizer'):
            action_argmax = tf.argmax(self._action_ph, 1, name='action_argmax')
            action_onehot = tf.one_hot(action_argmax, self.env.action_space.shape[0],
                                       1.0, 0.0, name='action_onehot')
            q_selected = tf.reduce_sum(self.net.output * action_onehot, 1)
            td_error = self._reward_ph - q_selected
            loss = tf.reduce_mean(tf.square(td_error), name='loss')
            grads = self.global_agent.opt.gradients(loss, self._weights)
            self._train_op = self.global_agent.opt.apply_gradients(grads, self.global_agent.weights)
            self._sync_op = [local_w.assign(self.global_agent.weights[i])
                             for i, local_w in enumerate(self._weights)]
        with tf.variable_scope(self._scope):
            add_observation_summary(self.net.input_ph, self.env)
            tf.summary.histogram('action', action_onehot)
            tf.summary.histogram('action_values', self.net.output)
            tf.summary.scalar('loss', loss)
        self._summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                              self._scope))

    def train_on_batch(self, obs, actions, rewards, obs_next, term, summarize=False):
        expected_reward = 0
        if not term:
            expected_reward = np.max(self.global_agent.target_predict(obs_next))
            self._ep_q.add(expected_reward)
        else:
            self._ep_reward.add(self._reward_accum)
            self._reward_accum = 0
        rewards = discount_rewards(rewards, self.gamma, expected_reward)
        _, summary = self.sess.run([self._train_op, self._summary_op if summarize else self._no_op],
                                   feed_dict={
                                       self.net.input_ph: obs,
                                       self._action_ph: actions,
                                       self._reward_ph: rewards
                                   })
        return summary

    def predict_action(self, obs, policy=None, step=0):
        action_values = self.predict_on_batch([obs])
        return self.policy.select_action(self.env, action_values, step)
