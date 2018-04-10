from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from six.moves import range

from reinforceflow.agents.agent import BaseAgent
from reinforceflow.core import losses, Continuous, Tuple
from reinforceflow.core.optimizer import Optimizer, RMSProp
from reinforceflow.core.policy import GreedyPolicy
from reinforceflow.utils import tensor_utils, utils


class ActorCritic(BaseAgent):
    def __init__(self, env, model, use_double=True, restore_from=None, device='/gpu:0',
                 optimizer=None, additional_losses=set(), trajectory_batch=True,
                 trainable_weights=None, saver_keep=3, name='ActorCritic'):
        """Constructs Deep Q-Learning agent.
         Includes the following implementations:
            1. Human-level control through deep reinforcement learning, Mnih et al., 2015.
            2. Dueling Network Architectures for Deep Reinforcement Learning, Wang et al., 2015.
                See `models.DeepQModel`.
            3. Deep Reinforcement Learning with Double Q-learning, Hasselt et al., 2016.
                See `use_double` argument.
            4. Prioritized Experience Replay, Schaul et al., 2015.
                See `core.replay.ProportionalReplay`.

        See `core.BaseDeepQ`.
        Args:
            env (gym.Env): Environment instance.
            model (models.AbstractFactory): Network factory.
            use_double (bool): Enables Double DQN.
            restore_from (str): Path to the pre-trained model.
            device (str): TensorFlow device, used for graph creation.
            optimizer (str or Optimizer): Agent's optimizer.
                By default: RMSProp(lr=2.5e-4, momentum=0.95).
            additional_losses (set): Set of additional losses.
            trainable_weights (list): List of trainable weights.
                Network architecture must be exactly the same as provided for this agent.
                If provided, current agent weights will remain constant.
                Pass None, to optimize current agent network.
            saver_keep (int): Maximum number of checkpoints can be stored at once.
        """
        super(ActorCritic, self).__init__(env=env, model=model, device=device,
                                          saver_keep=saver_keep, name=name)
        self.use_double = use_double
        self.trajectory_batch = trajectory_batch

        if isinstance(self.env.action_space, Continuous):
            raise ValueError('%s does not support environments with continuous '
                             'action space.' % self.__class__.__name__)
        if isinstance(self.env.action_space, Tuple):
            raise ValueError('%s does not support environments with multiple '
                             'action spaces.' % self.__class__.__name__)

        self._last_log_time = time.time()
        self._last_target_sync = self.step

        with tf.device(self.device):
            self.terms = tf.placeholder('bool', [None], name='term')
            self.traj_ends = tf.placeholder('bool', [None], name='trajectory')
            self.gamma = tf.placeholder('float32', [], name='gamma')
            self.obs_next = tf.placeholder('float32',
                                           shape=[None] + list(self.env.observation_space.shape),
                                           name='obs_next')

            with tf.variable_scope(self._scope + 'optimizer'):
                bootstrap_idx = tf.logical_xor(self.traj_ends, self.terms)
                obs_next = tf.cond(tf.greater(tf.reduce_sum(tf.cast(bootstrap_idx, 'int32')), 0),
                                   lambda: tf.boolean_mask(self.obs_next, bootstrap_idx),
                                   lambda: self.obs_next)

            with tf.variable_scope(self._scope + 'network', reuse=True):
                self.ev_net = self.model.build_from_inputs(inputs=obs_next,
                                                           output_space=self.env.action_space)

            with tf.variable_scope(self._scope + 'optimizer'):
                discount = tensor_utils.discount_trajectory_op(self.rewards,
                                                               self.terms,
                                                               self.traj_ends,
                                                               self.gamma,
                                                               self.ev_net['value'])
                compositor = losses.LossCompositor()
                compositor.add(losses.PolicyGradientLoss(coef=1.0))
                compositor.add(losses.EntropyLoss(coef=-0.01))
                compositor.add(losses.TDLoss(coef=1.0))
                compositor.add(additional_losses)
                loss = compositor.loss(endpoints=self.net,
                                       action=self.actions,
                                       reward=discount,
                                       term=self.terms)
                if optimizer is None:
                    self.opt = RMSProp(7e-4, decay=0.99, epsilon=0.1)
                self.opt = Optimizer.create(optimizer, self.global_step)
                trainable_weights = self.weights if trainable_weights is None else trainable_weights

                grads = self.opt.gradients(loss, self.weights)
                self._train_op = {'value': self.net['value'],
                                  'target': discount,
                                  'loss': loss,
                                  'minimize': self.opt.apply_gradients(grads, trainable_weights)}

        with tf.variable_scope(self._scope) as sc:
            tensor_utils.add_observation_summary(self.net['input'], self.env)
            tf.summary.histogram('agent/action', self.actions)
            tf.summary.scalar('agent/learning_rate', self.opt.lr_ph)
            tf.summary.scalar('metrics/loss', loss)
            tf.summary.scalar('metrics/avg_Q', tf.reduce_mean(self.ev_net['value']))
            tf.summary.scalar('value', tf.reduce_mean(self.net['value']))
            tf.summary.scalar('lr', self.opt.lr_ph)
            tf.summary.scalar('loss', loss)
            self._summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, sc.name))
        self.sess.run(tf.global_variables_initializer())
        if restore_from and tf.train.latest_checkpoint(restore_from):
            self.load_weights(restore_from)

    def predict_on_batch(self, obs_batch):
        """Computes action-values for given batch of observations."""
        return self.sess.run(self.net['policy'], {self.net['input']: obs_batch})

    def act(self, obs):
        """Computes action with maximum probability."""
        action_probs = self.predict_on_batch([obs])
        return GreedyPolicy.select_action(self.env, action_probs)

    def explore(self, obs, step=0):
        """Computes action with given exploration policy for given observation."""
        action_probs = self.predict_on_batch([obs])
        act = np.random.choice(range(len(action_probs[0])), p=action_probs[0])
        return utils.onehot(act, self.env.action_space.shape)

    def train_on_batch(self, obs, actions, rewards, term, obs_next, traj_ends,
                       lr, gamma=0.99, summarize=False, importance=None):
        self._train_op['summary'] = self._summary_op if summarize else self._no_op
        return self.sess.run(self._train_op, {self.net['input']: obs,
                                              self.actions: actions,
                                              self.rewards: rewards,
                                              self.terms: term,
                                              self.obs_next: obs_next,
                                              self.traj_ends: traj_ends,
                                              self.opt.lr_ph: lr,
                                              self.gamma: gamma
                                              })
