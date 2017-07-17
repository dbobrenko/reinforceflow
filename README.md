# ReinforceFlow

**ReinforceFlow** is an AI framework for training and evaluating **Reinforcement Learning** (RL) algorithms.

ReinforceFlow contains a bunch of implementations of the state-of-the-art solutions for RL problems.

ReinforceFlow was initially designed to work with [OpenAI Gym](https://gym.openai.com/) interface and [TensorFlow framework](https://www.tensorflow.org/) for some advanced deep learning stuff.

:construction: *Note: Project is under heavy development and sometimes could be unstable. Some of it's components may undergo considerable changes in future.*

## Requirements
  1. Python 2.7 or 3.4+;
  2. [OpenAI Gym](https://gym.openai.com/);
  3. `pip install -e .`


## Usage
Examples can be found at `examples` directory.

To train **Async N-step DQN on Breakout**, run:
```
python examples/async_dqn_breakout.py
```

To train **Async N-step DQN on CartPole**, run:
```
python examples/async_dqn_cartpole.py
```

To train **DQN on CartPole**, run:
```
python examples/dqn_cartpole.py
```


## Constantly evolving tasks list:
  - [x] [DQN] Deep Q-Network ([implementation](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/dqn.py))
  - [x] Double Deep Q-Network ([implementation](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/dqn.py))
  - [x] Asynchronous N-step Q-Learning ([implementation](https://github.com/dbobrenko/reinforceflow/blob/master/reinforceflow/agents/async_dqn.py))
  - [ ] Prioritized Replay
  - [ ] [A3C] Asynchronous Advantage Actor-Critic
  - [ ] Dueling Deep Q-Network
  - [ ] Stochastic PG


## Related projects:
  - [OpenAI Baselines](https://github.com/openai/baselines)
  - [OpenAI rllab](https://github.com/openai/rllab)
  - [Keras-rl](https://github.com/matthiasplappert/keras-rl)
