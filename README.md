# ReinforceFlow
[![Build Status](https://travis-ci.org/dbobrenko/reinforceflow.svg?branch=master)](https://travis-ci.org/dbobrenko/reinforceflow)

**ReinforceFlow** is an AI framework for training and evaluating **Reinforcement Learning** (RL) algorithms.

ReinforceFlow was initially designed to work with [OpenAI Gym](https://gym.openai.com/) interface and [TensorFlow framework](https://www.tensorflow.org/) for some advanced deep learning stuff.

:blue_book: **Documentation coming soon.**

## Requirements
  1. [OpenAI Gym](https://gym.openai.com/);
  2. (Installs during setup) TensorFlow >= 1.0.0
  3. (Optional) For more environments, you can install [OpenAI Universe](https://universe.openai.com/);

## Installation
  1. `git clone https://github.com/dbobrenko/reinforceflow.git`
  2. `pip install -e .[tf-gpu]`

     In case if you have no CUDA device available, use CPU-only TensorFlow:

     `pip install -e .[tf]`


## Usage
Examples of usage can be found at `examples` directory.

To train **Async N-step DQN on SpaceInvaders**, run:
```
python examples/async_dqn/adqn_spaceinv.py
```

To train **Async N-step DQN on Pong**, run:
```
python examples/async_dqn/adqn_pong.py
```

To train **Async N-step DQN on CartPole**, run:
```
python examples/async_dqn/adqn_cartpole.py
```

To train **Double DQN on CartPole**, run:
```
python examples/double_dqn/dqn_cartpole.py
```

To train **Dueling DQN on CartPole**, run:
```
python examples/dueling_dqn/dqn_cartpole.py
```


## Constantly evolving tasks list:
  - [x] Deep Q-Network ([implementation](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/dqn.py))
  - [x] Double Deep Q-Network ([implementation](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/dqn.py))
  - [x] Dueling Deep Q-Network ([implementation](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/nets.py))
  - [x] Asynchronous N-step Q-Learning ([implementation](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/async_dqn.py))
  - [x] Prioritized Proportional Experience Replay ([in progress](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/core/replay.py))
  - [ ] [A3C] Asynchronous Advantage Actor-Critic
  - [ ] Stochastic Policy Gradients


## Related projects:
  - [OpenAI Baselines](https://github.com/openai/baselines)
  - [Keras-rl](https://github.com/matthiasplappert/keras-rl)
  - [OpenAI rllab](https://github.com/openai/rllab)
