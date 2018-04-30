from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from reinforceflow.agents.agent import BaseAgent
from reinforceflow.core import losses, Continuous, Tuple
from reinforceflow.core.optimizer import Optimizer, RMSProp
from reinforceflow.core.policy import GreedyPolicy
from reinforceflow.utils import tensor_utils


class DeepQ(BaseAgent):
    def __init__(self, env, model, use_double=True, restore_from=None, device='/gpu:0',
                 optimizer=None, policy=None, targetfreq=40000, additional_losses=set(),
                 trainable_weights=None, target_net=None, target_weights=None,
                 saver_keep=3, name='DeepQ'):
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
            targetfreq (int): Target network update frequency(in seen observations).
            additional_losses (set): Set of additional losses.
            trainable_weights (list): List of trainable weights.
                Network architecture must be exactly the same as provided for this agent.
                If provided, current agent weights will remain constant.
                Pass None, to optimize current agent network.
            target_net (Network): Custom target network. Disables target sync.
                Pass None, to use agent's target network.
            saver_keep (int): Maximum number of checkpoints can be stored at once.
        """
        super(DeepQ, self).__init__(env=env, model=model, device=device,
                                    saver_keep=saver_keep, name=name)
        self.use_double = use_double
        self._target_freq = targetfreq
        self.policy = policy
        self._last_target_sync = self.step

        if isinstance(self.env.action_space, Continuous):
            raise ValueError('%s does not support environments with continuous '
                             'action space.' % self.__class__.__name__)
        if isinstance(self.env.action_space, Tuple):
            raise ValueError('%s does not support environments with multiple '
                             'action spaces.' % self.__class__.__name__)
        self.target_net = target_net
        self.target_weights = target_weights
        if target_net is None:
            with tf.variable_scope(self._scope + 'target_network') as scope:
                self.target_net = self.model.build(input_space=self.env.observation_space,
                                                   output_space=self.env.action_space)
                self.target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                        scope.name)
        self._target_update = [w.assign(self.weights[i])
                               for i, w in enumerate(self.target_weights)]
        if self.target_weights is None:
            raise ValueError("Target network weights must be provided with target network.")
        self.target_net['last_sync'] = self.step

        if optimizer is None:
            optimizer = RMSProp(0.00025, momentum=0.95, epsilon=0.01)
        with tf.device(self.device):
            self.importance = tf.placeholder('float32', [None], name='importance_sampling')
            self.terms = tf.placeholder('bool', [None], name='term')
            self.traj_ends = tf.placeholder('bool', [None], name='trajectory')
            self.gamma = tf.placeholder('float32', [], name='gamma')
            self.obs_next = tf.placeholder('float32',
                                           shape=[None] + list(self.env.observation_space.shape),
                                           name='obs_next')

            with tf.variable_scope(self._scope + 'network', reuse=True):
                self.ev_net = self.model.build_from_inputs(inputs=self.obs_next,
                                                           output_space=self.env.action_space)

            with tf.variable_scope(self._scope + 'optimizer'):
                if self.use_double:
                    q_idx = tf.argmax(self.ev_net['value'], 1)
                    q_onehot = tf.one_hot(q_idx, self.env.action_space.shape[0], 1.0)
                    q_next_max = tf.reduce_sum(self.target_net['value'] * q_onehot, 1)
                else:
                    q_next_max = tf.reduce_max(self.target_net['value'], 1)

                self.q_next_max = q_next_max
                with tf.variable_scope(self._scope + 'optimizer'):
                    target = tensor_utils.discount_trajectory_op(self.rewards,
                                                                 self.terms,
                                                                 self.traj_ends,
                                                                 self.gamma,
                                                                 q_next_max)
                    self.target = target

                qloss = losses.QLoss(1.0, importance_sampling=self.importance)
                compositor = losses.LossCompositor([qloss])
                compositor.add(additional_losses)
                loss = compositor.loss(endpoints=self.net,
                                       action=self.actions,
                                       reward=target,
                                       term=self.terms)
                self.opt = Optimizer.create(optimizer, self.global_step)
                trainable_weights = self.weights if trainable_weights is None else trainable_weights

                grads = self.opt.gradients(loss, self.weights)
                self._train_op = {'value': self.net['value'],
                                  'target': target,
                                  'minimize': self.opt.apply_gradients(grads, trainable_weights)}

        tensor_utils.add_observation_summary(self.net['input'], self.env)
        tf.summary.histogram('agent/action', self.actions)
        tf.summary.histogram('agent/action_values', self.net['value'])
        tf.summary.scalar('agent/learning_rate', self.opt.lr)
        tf.summary.scalar('metrics/loss', loss)
        tf.summary.scalar('metrics/avg_Q', tf.reduce_mean(q_next_max))
        self._summary_op = tf.summary.merge_all()
        self._saver = tf.train.Saver(max_to_keep=saver_keep)
        tensor_utils.initialize_variables(self.sess)
        if restore_from and tf.train.latest_checkpoint(restore_from):
            self.load_weights(restore_from)

    def predict_on_batch(self, obs_batch):
        """Computes action-values for given batch of observations."""
        return self.sess.run(self.net['value'], {self.net['input']: obs_batch})

    def act(self, obs):
        """Computes action with greedy policy for given observation."""
        action_values = self.predict_on_batch([obs])
        return GreedyPolicy.select_action(self.env, action_values)

    def explore(self, obs, step=None):
        action_values = self.predict_on_batch([obs])
        step = self.step if step is None else step
        return self.policy.select_action(self.env, action_values, step=step)

    def load_weights(self, checkpoint):
        super(DeepQ, self).load_weights(checkpoint)
        self.target_update()

    def target_predict(self, obs):
        """Computes target network action-values with for given batch of observations."""
        return self.sess.run(self.target_net['value'], {self.target_net['input']: obs})

    def target_update(self):
        """Syncs target network with behaviour network."""
        self.sess.run(self._target_update)

    def train_on_batch(self, obs, actions, rewards, term, obs_next, traj_ends,
                       lr, gamma=0.99, summarize=False, importance=None):
        if self.step - self.target_net['last_sync'] > self._target_freq:
            self.target_net['last_sync'] = self.step
            self.target_update()

        if importance is None:
            importance = np.ones_like(rewards)

        self._train_op['summary'] = self._summary_op if summarize else self._no_op
        return self.sess.run(self._train_op, {self.net['input']: obs,
                                              self.actions: actions,
                                              self.rewards: rewards,
                                              self.terms: term,
                                              self.target_net['input']: obs_next,
                                              self.obs_next: obs_next,
                                              self.traj_ends: traj_ends,
                                              self.importance: importance,
                                              self.opt.lr_ph: lr,
                                              self.gamma: gamma
                                              })
