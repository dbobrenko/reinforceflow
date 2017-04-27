from gym.envs.registration import registry, register, make, spec
from .gridworld import *

nondeterministic = False
register(
    id='GridWorld-v0',
    entry_point='gridworld:GridWorld',
    kwargs={'game_name': 'GridWorld', 'display_screen': False},
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
    nondeterministic=nondeterministic,
)
