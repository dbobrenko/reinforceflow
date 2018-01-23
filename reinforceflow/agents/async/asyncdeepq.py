from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from reinforceflow.agents.async.base_async import AsyncAgent, AsyncThreadAgent
from reinforceflow.core import AgentCallback
from reinforceflow.core import losses
from reinforceflow.core.agent import BaseDeepQ
from reinforceflow.core.optimizer import RMSProp
from reinforceflow.utils import discount
from reinforceflow.utils.tensor_utils import add_observation_summary


class _AsyncDQNCallback(AgentCallback):
    def __init__(self):
        self.last_target_update = 0

    def on_iter_end(self, agent, logs):
        if agent.step - self.last_target_update >= agent.target_freq:
            self.last_target_update = agent.step
            agent.target_update()


class AsyncDeepQ(BaseDeepQ, AsyncAgent):
    """Constructs Asynchronous N-step Q-Learning agent, based on paper:
    Asynchronous Methods for Deep Reinforcement Learning, Mnih et al., 2016
    (https://arxiv.org/abs/1602.01783v2).

    See `AsyncAgent`, `BaseDeepQ`.
    Args:
        restore_from (str): Path to the pre-trained model.
    """
    def __init__(self, env, model, restore_from=None, device='/gpu:0', optimizer=None,
                 num_threads=8, saver_keep=3, name='AsyncDeepQ'):
        if optimizer is None:
            optimizer = RMSProp(7e-4, decay=0.99, epsilon=0.1)
        super(AsyncDeepQ, self).__init__(env=env,
                                         model=model,
                                         thread_agent=_ThreadDeepQ,
                                         device=device,
                                         optimizer=optimizer,
                                         num_threads=num_threads,
                                         saver_keep=saver_keep,
                                         name=name)
        self.target_freq = None
        self.sess.run(tf.global_variables_initializer())
        if restore_from is not None and tf.train.latest_checkpoint(restore_from) is not None:
            self.load_weights(restore_from)

    def train(self, maxsteps, policy, log_dir, log_freq, batch_size=5, gamma=0.99, lr_schedule=None,
              target_freq=10000, log_on_term=True, render=False, test_env=None, test_render=False,
              test_episodes=1, test_maxsteps=5000, callbacks=set()):
        """See `AsyncAgent.train` and `DeepQ.train`."""
        self.target_freq = target_freq
        callbacks = set(callbacks)
        callbacks.add(_AsyncDQNCallback())
        super(AsyncDeepQ, self).train(maxsteps, policy, log_dir,
                                      log_freq=log_freq,
                                      batch_size=batch_size,
                                      gamma=gamma,
                                      lr_schedule=lr_schedule,
                                      log_on_term=log_on_term,
                                      render=render,
                                      test_env=test_env,
                                      test_render=test_render,
                                      test_episodes=test_episodes,
                                      callbacks=callbacks)


class _ThreadDeepQ(BaseDeepQ, AsyncThreadAgent):
    def __init__(self, env, model, global_agent, device, name=''):
        super(_ThreadDeepQ, self).__init__(env=env, model=model, global_agent=global_agent,
                                           device=device, name=name)
        with tf.device(self.device), tf.variable_scope(self._scope + 'optimizer'):
            loss, _ = losses.td_error_q(q_logits=self.net['out'],
                                        action=self._action_ph,
                                        target=self._reward_ph,
                                        name='loss')
            grads = self.global_agent.opt.gradients(loss, self._weights)
            self._train_op = self.global_agent.opt.apply_gradients(grads, self.global_agent.weights)
            self._sync_op = [local_w.assign(self.global_agent.weights[i])
                             for i, local_w in enumerate(self._weights)]

        with tf.variable_scope(self._scope):
            add_observation_summary(self.net['in'], self.env)
            tf.summary.histogram('action_values', self.net['out'])
            tf.summary.histogram('action', self._action_ph)
            tf.summary.scalar('lr', self.global_agent.opt.lr_ph)
            tf.summary.scalar('loss', loss)
        self._summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                              self._scope))

    def train_on_batch(self, obs, actions, rewards, obs_next, term,
                       lr, gamma=0.99, summarize=False):
        expected_reward = (1-term) * np.max(self.global_agent.target_predict(obs_next))
        rewards = discount(rewards, gamma, expected_reward)
        _, summary = self.sess.run([self._train_op, self._summary_op if summarize else self._no_op],
                                   feed_dict={
                                       self.net['in']: obs,
                                       self._action_ph: actions,
                                       self._reward_ph: rewards,
                                       self.global_agent.opt.lr_ph: lr
                                   })
        return summary
