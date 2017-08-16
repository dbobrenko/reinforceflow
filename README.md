# ReinforceFlow

**ReinforceFlow** is an AI framework for training and evaluating **Reinforcement Learning** (RL) algorithms.

ReinforceFlow contains a bunch of implementations of the state-of-the-art solutions for RL problems.

ReinforceFlow was initially designed to work with [OpenAI Gym](https://gym.openai.com/) interface and [TensorFlow framework](https://www.tensorflow.org/) for some advanced deep learning stuff. However, there are plans to support more environments in future, as well as custom-made.

:construction: *Note: Project is under heavy development and sometimes it could be unstable. Some of it's components may undergo considerable changes in future.*

## Requirements & Installation
  1. Python 2.7 or 3.4+;
  2. [OpenAI Gym](https://gym.openai.com/);
  3. `pip install -e .`


## Usage
Examples of usage can be found at `examples` directory.

To train **Async N-step DQN on SpaceInvaders**, run:
```
python examples/async_dqn/adqn_spaceinv.py
```

To train **Async N-step DQN on CartPole**, run:
```
python examples/async_dqn/adqn_cartpole.py
```

To train **Double DQN on CartPole**, run:
```
python examples/double_dqn/dqn_cartpole.py
```

To train **Double DQN on Pong**, run:
```
python examples/double_dqn/dqn_pong.py
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
  - [OpenAI rllab](https://github.com/openai/rllab)
  - [Keras-rl](https://github.com/matthiasplappert/keras-rl)
