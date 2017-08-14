from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import numpy.testing as npt
from reinforceflow.core import ExperienceReplay, ProportionalReplay
# TODO: add tests for ProportionalReplay.update


def test_replay_add():
    cap = 500000
    replay = ExperienceReplay(capacity=cap, min_size=500, batch_size=32)
    for i in range(3*cap):
        replay.add(0, 0, 0, 0, False)
    assert replay.size == cap


def test_replay_sample():
    cap = 256
    batch_size = 32
    replay = ExperienceReplay(capacity=cap, min_size=cap, batch_size=batch_size)
    for i in range(0, 10*cap, 10):
        replay.add(obs=i, action=i+1, reward=i+2, obs_next=i+10, term=i % 20)
    for _ in range(10):
        obs, action, reward, obs_next, term, idx, importance = replay.sample()
        for o, a, r, o_next, t, i in zip(obs, action, reward, obs_next, term, idx):
            assert a - 1 == o
            assert r - 2 == o
            assert o_next - 10 == o
            assert t == o % 20
            assert i == o // 10
    assert len(obs) == batch_size


def test_replay_sample_term():
    cap = 1024
    batch_size = 128
    replay = ExperienceReplay(capacity=cap, min_size=cap, batch_size=batch_size)
    for i in range(cap):
        replay.add(obs=i, action=0, reward=0, obs_next=i+1, term=i % 5 == 0)
    obs, a, r, obs_next, terms, idxs, importance = replay.sample()
    assert len(obs) == batch_size
    assert len(obs_next) == batch_size
    assert len(a) == batch_size
    assert len(r) == batch_size
    assert len(terms) == batch_size
    assert len(idxs) == batch_size
    for o, o_next, term in zip(obs, obs_next, terms):
        if not term:
            assert o+1 == o_next


def test_proportional_add():
    cap = 500000
    replay = ProportionalReplay(capacity=cap, min_size=500, batch_size=32, alpha=1.0)
    for i in range(3*cap):
        replay.add(0, 0, 0, 0, False)
    assert replay.size == cap


def test_prop_replay_sample():
    cap = 512
    batch_size = 256
    replay = ProportionalReplay(capacity=cap, min_size=cap, batch_size=batch_size, alpha=1.0)
    for i in range(2*cap):
        replay.add(obs=i, action=0, reward=0, obs_next=i+1, term=False, priority=i)
    obs, a, r, obs_next, terms, idxs, importance = replay.sample()
    assert len(obs) == batch_size
    assert len(obs_next) == batch_size
    assert len(a) == batch_size
    assert len(r) == batch_size
    assert len(terms) == batch_size
    assert len(idxs) == batch_size
    npt.assert_equal([i+1 for i in obs], obs_next)


def test_prop_replay_distribution():
    priors = np.array([20000.0, 30000.0, 500.0, 49500.0, 0.0])
    cap = 2048
    batch_size = 32
    replay = ProportionalReplay(capacity=cap, min_size=cap, batch_size=batch_size, alpha=1.0)
    s = int(np.sum(priors))
    expected_priors = priors / s
    received_priors = dict.fromkeys(priors.tolist(), 0)
    for p in priors:
        replay.add(obs=p, action=0, reward=0, obs_next=0, term=False, priority=p)
    for i in range(1000):
        obs, a, r, obs_next, terms, idxs, importance = replay.sample()
        for o in obs:
            received_priors[float(o)] += 1
    received_priors = np.asarray(list(received_priors)) / s
    npt.assert_almost_equal(expected_priors, received_priors, decimal=4)


def test_prop_replay_update():
    priors = np.array([20000.0, 30000.0, 500.0, 49500.0, 0.0])
    cap = 2048
    batch_size = 32
    replay = ProportionalReplay(capacity=cap, min_size=cap, batch_size=batch_size, alpha=1.0)
    s = int(np.sum(priors))
    expected_priors = priors / s
    received_priors = dict.fromkeys(priors.tolist(), 0)
    for p in priors:
        replay.add(obs=p, action=0, reward=0, obs_next=0, term=False)
    replay.update(list(range(1000)), priors)
    for i in range(1000):
        obs, a, r, obs_next, terms, idxs, importance = replay.sample()
        for o in obs:
            received_priors[float(o)] += 1
    received_priors = np.asarray(list(received_priors)) / s
    npt.assert_almost_equal(expected_priors, received_priors, decimal=4)
