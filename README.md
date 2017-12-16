# ReinforceFlow
[![Build Status](https://travis-ci.org/dbobrenko/reinforceflow.svg?branch=master)](https://travis-ci.org/dbobrenko/reinforceflow)

A framework for training and evaluating **Reinforcement Learning** (RL) algorithms.
Built with Python, [TensorFlow framework](https://www.tensorflow.org/) and [OpenAI Gym](https://gym.openai.com/) interface.

:construction: Currently under heavy development and some of its components may suffer from instability issues.

:blue_book: **Documentation coming soon.**

## Installation
  1. `git clone https://github.com/dbobrenko/reinforceflow.git`
  2. `cd reinforceflow`
  3. `pip install -e .[tf-gpu]`

     In case if you have no CUDA device available, use CPU-only TensorFlow:

     `pip install -e .[tf]`

  3. To get examples working, install Gym by following the instructions at [OpenAI Gym repo](https://github.com/openai/gym);
  4. (Optional) For more environments, you can install [OpenAI Universe](https://universe.openai.com/);



## Usage
Examples can be found in `examples` directory:
```
# To train A3C on CartPole:
python examples/a3c/a3c_carpole.py

# To train DQN on CartPole:
python examples/dqn/dqn_cartpole.py

# To train DQN w/ Prior Expeience on Pong:
python examples/dqn/dqn_pong.py
```


## Constantly evolving tasks list:
  - [x] **DQN**: [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) ([**implementation**](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/dqn.py))
  - [x] **Double DQN**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) ([**implementation**](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/dqn.py))
  - [x] **Dueling DQN** [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) ([**implementation**](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/dqn.py))
  - [x] [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952v4) ([**implementation**](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/core/replay.py))
  - [x] **Async DQN**: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2) ([**implementation**](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/async/adqn.py))
  - [x] **A3C**: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2) ([**implementation**](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/async/a3c.py))
  - [ ] Evolution Strategies


## Related projects:
  - [OpenAI Baselines](https://github.com/openai/baselines)
  - [Keras-rl](https://github.com/matthiasplappert/keras-rl)
  - [OpenAI rllab](https://github.com/openai/rllab)
