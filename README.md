# ReinforceFlow
[![Build Status](https://travis-ci.org/dbobrenko/reinforceflow.svg?branch=master)](https://travis-ci.org/dbobrenko/reinforceflow)

**ReinforceFlow** is an AI framework for training and evaluating **Reinforcement Learning** (RL) algorithms.

ReinforceFlow was initially designed to work with [OpenAI Gym](https://gym.openai.com/) and [OpenAI Universe](https://universe.openai.com/) interface, using [TensorFlow framework](https://www.tensorflow.org/). However, other interfaces, as well as custom environment support, will be added soon.

:blue_book: **Documentation coming soon.**

## Requirements
  1. [OpenAI Gym](https://gym.openai.com/);
  2. (Installs during setup) TensorFlow >= 1.0.0;
  3. (Optional) For more environments, you can install [OpenAI Universe](https://universe.openai.com/);

## Installation
  1. `git clone https://github.com/dbobrenko/reinforceflow.git`
  2. `cd reinforceflow`
  3. `pip install -e .[tf-gpu]`

     In case if you have no CUDA device available, use CPU-only TensorFlow:

     `pip install -e .[tf]`


## Usage
All tested examples can be found in `examples` directory.

Some examples:
```
# To train Async N-step DQN on Pong:
python examples/async_dqn/adqn_pong.py

# To train Async N-step DQN on CartPole:
python examples/async_dqn/adqn_cartpole.py

# To train Double DQN on CartPole:
python examples/double_dqn/dqn_cartpole.py

# To train Dueling DQN on CartPole:
python examples/dueling_dqn/dqn_cartpole.py
```


## Constantly evolving tasks list:
  - [x] **DQN**: [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) ([**implementation**](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/dqn.py))
  - [x] **Double DQN**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) ([**implementation**](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/dqn.py))
  - [x] **Dueling DQN** [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) ([**implementation**](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/nets/dueling.py))
  - [x] **Async DQN**: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2) ([**implementation**](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/async_dqn.py))
  - [x] [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952v4) ([**implementation**](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/core/replay.py))
  - [x] **A3C**: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2) ([**in progress**](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/a3c.py))
  - [ ] Stochastic Policy Gradients
  - [ ] Evolution Strategies


## Related projects:
  - [OpenAI Baselines](https://github.com/openai/baselines)
  - [Keras-rl](https://github.com/matthiasplappert/keras-rl)
  - [OpenAI rllab](https://github.com/openai/rllab)
