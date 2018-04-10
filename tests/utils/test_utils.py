from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.testing as npt
import reinforceflow as rf
from reinforceflow.utils import utils


def test_reward_stats():
    steps = 10000
    episodes = 5
    step_reward = 1
    step = 0
    episode = 0
    total_rewards = [[step_reward] * steps for _ in range(episodes)]
    step_av = np.mean(total_rewards)
    episode_av = np.mean([np.sum(i) for i in total_rewards])
    stats = rf.core.stats.RewardStats()
    for ep in total_rewards:
        for i, r in enumerate(ep):
            term = i == steps-1
            step += 1
            episode += int(term)
            stats.add(reward=r, terminal=term)
    assert stats.step_average() == step_av
    assert stats.episode_average() == episode_av


def test_image_preprocess():
    init_h, init_w, init_c = 224, 156, 3
    image_rgb = np.ones((init_h, init_w, init_c), dtype='uint8')

    h, w = 80, 80
    gray = True
    resized_gray = utils.image_preprocess(image_rgb, h, w, to_gray=gray)
    assert resized_gray.shape == (h, w, 1 if gray else init_c)

    h, w = 48, 110
    gray = False
    resized = utils.image_preprocess(image_rgb, h, w, to_gray=gray)
    assert resized.shape == (h, w, 1 if gray else init_c)

    gray = False
    same = utils.image_preprocess(image_rgb, init_h, init_w, to_gray=gray)
    assert same.shape == (init_h, init_w, 1 if gray else init_c)


def test_isarray():
    assert utils.isarray([1, 2, 3])
    assert utils.isarray((1, 2, 3))
    assert utils.isarray({1, 2, 3})
    assert utils.isarray(np.array([1, 2, 3]))


def _discount(rewards, gamma, ev):
    discount_sum = ev
    exp_rewards = [0] * len(rewards)
    for i in reversed(range(len(rewards))):
        discount_sum = rewards[i] + gamma * discount_sum
        exp_rewards[i] = discount_sum
    return exp_rewards


def test_discount_rewards():
    rewards = [1, 2, 3, 4]
    gamma = 0.5
    ev = 1.0

    exp_rewards = _discount(rewards, gamma, ev)
    disc_rewards = utils.discount_rewards(rewards, gamma, ev)
    npt.assert_almost_equal(disc_rewards, exp_rewards, decimal=2)


def test_discount_trajectory():
    gamma = 0.5
    rewards1 = [1, 2, 3, 4]
    rewards2 = [1, 2, 3, 4]
    rewards = rewards1 + rewards2
    ev1 = 10
    ev2 = 0
    ev = [0, 0, 0, ev1, 0, 0, 0, 10]
    terms = [False, False, False, False, False, False, False, True]
    traj_ends = [False, False, False, True, False, False, False, True]
    exp_rewards1 = _discount(rewards1, gamma, ev1)
    exp_rewards2 = _discount(rewards2, gamma, ev2)
    exp_rewards = exp_rewards1 + exp_rewards2
    disc_rewards = utils.discount_trajectory(rewards, terms, traj_ends, gamma, ev)
    npt.assert_almost_equal(disc_rewards, exp_rewards, decimal=2)
