# ReinforceFlow

**ReinforceFlow** is an AI framework for training and evaluating **Reinforcement Learning** (RL) algorithms.

ReinforceFlow expects to contain high-quality realizations of the state-of-the-art solutions for RL problems.

ReinforceFlow was designed to work with [OpenAI Gym](https://gym.openai.com/) interface and [TensorFlow framework](https://www.tensorflow.org/) for some advanced deep learning stuff.

:construction: *Project is under heavy development and sometimes may be unstable. Some of it's components may undergo considerable changes in future.*

## Requirements
  1. Python 2.7 or 3.4+;
  2. `pip install -r requirements.txt`
  3. [OpenAI Gym](https://gym.openai.com/).
  
*ReinforceFlow as standalone library haven't been released yet.*

## Usage
Examples can be found at `examples` directory.

To train **Async N-step DQN on SpaceInvaders**, run:
```
python examples/async_dqn_atari.py
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
  - [x] Vanilla DQN
  - [x] Async DQN
  - [ ] Prioritized Replay
  - [ ] A3C
  - [ ] Dueling DQN
  - [ ] Stochastic PG


## Related projects:
  - [OpenAI Baselines](https://github.com/openai/baselines)
  - [OpenAI rllab](https://github.com/openai/rllab)
  - [Keras-rl](https://github.com/matthiasplappert/keras-rl)
