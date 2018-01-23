from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

EPSILON = 1e-8


def pg_loss(policy_logits, action, baseline, name='pg'):
    """Policy Gradient loss.

    Args:
        policy_logits (Tensor): Policy logits, i.e action probabilities.
        action (Placeholder): Vectorized action placeholder.
        baseline (Tensor): Policy baseline.
        name (str): Loss name.

    Returns (Tensor):
        Policy Gradient loss operation.
    """
    logprob = tf.log(policy_logits + EPSILON)
    return -tf.reduce_sum(tf.reduce_sum(logprob * action, 1) * baseline, name=name)


def pg_entropy_loss(policy_logits, action, baseline, coef_entropy=0.01, name=''):
    logprob = tf.log(policy_logits + EPSILON)
    pg = -tf.reduce_sum(tf.reduce_sum(logprob * action, 1) * baseline, name=name)
    entropy = -tf.reduce_sum(policy_logits * logprob)
    return pg - coef_entropy * entropy


def pg_entropy(policy_logits, name=''):
    """Policy Gradient Entropy.

    Args:
        policy_logits (Tensor): Policy logits, i.e action probabilities.
        name (str): Loss name.

    Returns (Tensor):
        Policy Gradient Entropy.
    """
    logprob = tf.log(policy_logits + EPSILON)
    return -tf.reduce_sum(policy_logits * logprob, name=name)


def advantage(value_logits, target, name=''):
    return tf.subtract(target, value_logits, name=name)


def td_error(value_logits, target, weights=None, name=''):
    """Temporal-Difference error.

    Args:
        value_logits: Value logits.
        target: Target value.
        weights: Weights applied to TD error (e.g. importance sampling).
        name (str): Loss name.

    Returns (tuple):
        A tuple of: TD error loss, target and logits difference.
    """
    td = target - value_logits
    td_weighted = td * weights if weights is not None else td
    return tf.reduce_mean(tf.square(td_weighted), name=name), td


def td_error_q(q_logits, action, target, weights=None, name=''):
    """Action-Value Temporal-Difference error. See `td_error`."""
    value = tf.reduce_sum(q_logits * action, 1)
    return td_error(value, target, weights, name=name)


def huber_loss(x):
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
