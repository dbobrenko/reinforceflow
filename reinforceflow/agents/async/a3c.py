from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from threading import Thread

import tensorflow as tf

from reinforceflow.agents.async.base_async import AsyncAgent, AsyncThreadAgent
from reinforceflow.core.optimizer import RMSProp
from reinforceflow.utils import discount_rewards
from reinforceflow.utils.tensor_utils import add_observation_summary


class A3CAgent(AsyncAgent):
    """Constructs Asynchronous Advantage Actor-Critic agent, based on paper:
    "Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016.
    (https://arxiv.org/abs/1602.01783v2)

    See `AsyncAgent`, `BaseDQNAgent`.
    Args:
        restore_from (str): Path to the pre-trained model.
    """
    def __init__(self, env, net_factory, restore_from=None, device='/gpu:0', steps=int(80e6),
                 optimizer=None, policy=None, num_threads=8, batch_size=5,
                 gamma=0.99, saver_keep=3, log_every_sec=300, name='A3CAgent'):
        if optimizer is None:
            optimizer = RMSProp(7e-4, decay=0.99, epsilon=0.1, lr_decay='linear')
        super(A3CAgent, self).__init__(env=env,
                                       net_factory=net_factory,
                                       thread_agent=_ThreadA3CAgent,
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
        self.sess.run(tf.global_variables_initializer())
        if restore_from and tf.train.latest_checkpoint(restore_from):
            self.load_weights(restore_from)


class _ThreadA3CAgent(AsyncThreadAgent, Thread):
    def __init__(self, env, net_factory, global_agent, policy, gamma,
                 batch_size, device, log_every_sec, name=''):
        super(_ThreadA3CAgent, self).__init__(env, net_factory, global_agent, policy, gamma,
                                              batch_size, device, log_every_sec, name)
        with tf.device(self.device), tf.variable_scope(self._scope + 'optimizer'):
            action_argmax = tf.argmax(self._action_ph, 1, name='action_argmax')
            action_onehot = tf.one_hot(action_argmax, self.env.action_space.shape[0],
                                       1.0, 0.0, name='action_one_hot')
            # Shape: BxA (Batch shape * Action shape)
            logprob = tf.log(self.net.output_policy + 1e-8)
            # Shape: B
            adv = self._reward_ph - self.net.output_value
            # Shape: sum(B*B) = 1
            _sum = tf.reduce_sum
            loss_policy = -_sum(_sum(logprob * action_onehot, 1) * tf.stop_gradient(adv))
            # Shape: sum(B) = 1
            loss_value = 0.5 * _sum(tf.square(adv))
            # Shape: sum(BxA * BxA) = 1
            entropy = 0.01 * _sum(self.net.output_policy * logprob)

            loss = loss_policy + loss_value + entropy
            grads = self.global_agent.opt.gradients(loss, self._weights)
            self._train_op = self.global_agent.opt.apply_gradients(grads,
                                                                   self.global_agent.weights)
            self._sync_op = [local_w.assign(self.global_agent.weights[i])
                             for i, local_w in enumerate(self._weights)]
        with tf.variable_scope(self._scope) as sc:
            add_observation_summary(self.net.input_ph, self.env)
            tf.summary.histogram('output_policy', self.net.output)
            tf.summary.histogram('policy_log_prob', logprob)
            tf.summary.scalar('output_value', tf.reduce_mean(self.net.output_value))
            tf.summary.scalar('advantage', tf.reduce_mean(adv))
            tf.summary.scalar('loss_policy', loss_policy)
            tf.summary.scalar('loss_value', loss_value)
            tf.summary.scalar('entropy', entropy)
            tf.summary.scalar('loss', loss)
            self._summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, sc.name))

    def train_on_batch(self, obs, actions, rewards, obs_next, term, summarize=False):
        expected_value = 0
        if not term:
            expected_value = self.sess.run(self.net.output_value, {self.net.input_ph: obs_next})
            self._q_stats.add(expected_value, term)
        rewards = discount_rewards(rewards, self.gamma, expected_value)
        _, summary = self.sess.run([self._train_op, self._summary_op if summarize else self._no_op],
                                   feed_dict={
                                       self.net.input_ph: obs,
                                       self._action_ph: actions,
                                       self._reward_ph: rewards
                                   })
        return summary
