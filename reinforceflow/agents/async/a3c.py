from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from threading import Thread

import tensorflow as tf

from reinforceflow.agents.async.base_async import AsyncAgent, AsyncThreadAgent
from reinforceflow.core.optimizer import RMSProp
from reinforceflow.core import losses
from reinforceflow.utils import discount
from reinforceflow.utils.tensor_utils import add_observation_summary


class A3C(AsyncAgent):
    """Constructs Asynchronous Advantage Actor-Critic agent, based on paper:
    "Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016.
    (https://arxiv.org/abs/1602.01783v2)

    See `AsyncAgent`, `BaseDeepQ`.
    Args:
        restore_from (str): Path to the pre-trained model.
    """
    def __init__(self, env, model, restore_from=None, device='/gpu:0', optimizer=None,
                 num_threads=8, saver_keep=3, name='A3C'):
        if optimizer is None:
            optimizer = RMSProp(7e-4, decay=0.99, epsilon=0.1)
        super(A3C, self).__init__(env=env,
                                  model=model,
                                  thread_agent=_ThreadA3C,
                                  device=device,
                                  optimizer=optimizer,
                                  num_threads=num_threads,
                                  saver_keep=saver_keep,
                                  name=name)
        self.sess.run(tf.global_variables_initializer())
        if restore_from and tf.train.latest_checkpoint(restore_from):
            self.load_weights(restore_from)


class _ThreadA3C(AsyncThreadAgent, Thread):
    def __init__(self, env, model, global_agent, device,
                 coef_policy=1.0, coef_value=0.5, coef_entropy=0.01, name=''):
        super(_ThreadA3C, self).__init__(env=env, model=model, global_agent=global_agent,
                                         device=device, name=name)
        with tf.device(self.device), tf.variable_scope(self._scope + 'optimizer'):
            # Loss:
            advantage = losses.advantage(value_logits=self.net['out_v'],
                                         target=self._reward_ph, name='advantage')
            loss_value = tf.reduce_sum(tf.square(advantage), name='value_loss')
            loss_value = coef_value * loss_value
            loss_policy = losses.pg_loss(policy_logits=self.net['out_pi'],
                                         action=self._action_ph,
                                         baseline=tf.stop_gradient(advantage),
                                         name='pg_loss')
            loss_policy = coef_policy * loss_policy
            entropy = losses.pg_entropy(self.net['out_pi'], name='entropy')
            entropy = coef_entropy * entropy
            loss = loss_policy + loss_value - entropy

            # Update:
            grads = self.global_agent.opt.gradients(loss, self._weights)
            self._train_op = self.global_agent.opt.apply_gradients(grads,
                                                                   self.global_agent.weights)
            self._sync_op = [local_w.assign(self.global_agent.weights[i])
                             for i, local_w in enumerate(self._weights)]

        with tf.variable_scope(self._scope) as sc:
            add_observation_summary(self.net['in'], self.env)
            tf.summary.scalar('value', tf.reduce_mean(self.net['out_v']))
            tf.summary.scalar('lr', self.global_agent.opt.lr_ph)
            tf.summary.scalar('advantage', tf.reduce_mean(advantage))
            tf.summary.scalar('loss_policy', loss_policy)
            tf.summary.scalar('loss_value', loss_value)
            tf.summary.scalar('entropy', entropy)
            tf.summary.scalar('loss', loss)
            self._summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, sc.name))

    def train_on_batch(self, obs, actions, rewards, obs_next, term,
                       lr, gamma=0.99, summarize=False):
        expected_value = (1-term) * self.sess.run(self.net['out_v'], {self.net['in']: obs_next})
        rewards = discount(rewards, gamma, expected_value)
        _, summary = self.sess.run([self._train_op, self._summary_op if summarize else self._no_op],
                                   feed_dict={
                                       self.net['in']: obs,
                                       self._action_ph: actions,
                                       self._reward_ph: rewards,
                                       self.global_agent.opt.lr_ph: lr
                                   })
        return summary
