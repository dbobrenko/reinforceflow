from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
from threading import Thread

import numpy as np

import reinforceflow.core.stats
from reinforceflow.utils import tensor_utils


def test_unit_rewards():
    steps = 10000
    episodes = 5
    step_reward = 1
    step = 0
    episode = 0
    on_term = True
    total_rewards = [[step_reward] * steps for i in range(episodes)]
    step_av = np.mean(total_rewards)
    episode_av = np.mean([np.sum(i) for i in total_rewards])
    summary = reinforceflow.core.stats.Stats(log_freq=None,
                                             log_on_term=on_term,
                                             log_performance=False)
    for ep in total_rewards:
        for i, r in enumerate(ep):
            term = i == steps-1
            step += 1
            episode += int(term)
            summary.add(reward=r, done=term, info={}, step=step, episode=episode)
    assert summary.reward_stats.step_average() == step_av
    assert summary.reward_stats.episode_average() == episode_av


def test_async_unit_rewards():
    steps = 10000
    episodes = 5
    step_reward = 1
    on_term = True
    num_threads = 8
    step = 0
    episode = 0
    lock = threading.Lock()
    thread_rewards = np.array([[step_reward] * steps for _ in range(episodes)])
    total_rewards = [thread_rewards.copy() for _ in range(num_threads)]
    step_av = np.mean(total_rewards)
    episode_av = np.mean([np.sum(i) for i in thread_rewards])
    summary = reinforceflow.core.stats.Stats(log_freq=None,
                                             log_on_term=on_term,
                                             log_performance=False)

    def thread(rewards, step, episode):
        for ep in rewards:
            for i, r in enumerate(ep):
                term = i == steps-1
                with lock:
                    step += 1
                    episode += int(term)
                summary.add(reward=r, done=term, info={}, step=step, episode=ep)
    threads = []
    for i in range(num_threads):
        t = Thread(target=thread, args=(total_rewards[i], step, episode))
        t.daemon = True
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    assert summary.reward_stats.step_average() == step_av
    assert summary.reward_stats.episode_average() == episode_av

test_async_unit_rewards()
test_unit_rewards()
