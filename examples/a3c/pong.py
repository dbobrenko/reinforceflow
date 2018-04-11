from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import reinforceflow
except ImportError:
    import os.path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    import reinforceflow
from reinforceflow.agents import ActorCritic
from reinforceflow.models import ActorCriticConv
from reinforceflow.core.optimizer import RMSProp
from reinforceflow.envs import AtariWrapper
from reinforceflow.trainers.async_trainer import AsyncTrainer

reinforceflow.set_random_seed(555)

env_name = "PongNoFrameskip-v4"
env = AtariWrapper(env_name,
                   action_repeat=4,
                   obs_stack=4,
                   new_width=84,
                   new_height=84,
                   to_gray=True,
                   noop_action=[1, 0, 0, 0, 0, 0],
                   start_action=[0, 1, 0, 0, 0, 0],
                   clip_rewards=True)
test_env = AtariWrapper(env_name,
                        action_repeat=4,
                        obs_stack=4,
                        new_width=84,
                        new_height=84,
                        to_gray=True,
                        start_action=[0, 1, 0, 0, 0, 0])

agent = ActorCritic(env,
                    model=ActorCriticConv(),
                    optimizer=RMSProp(7e-4, decay=0.99, epsilon=0.1))
threads = []
for i in range(16):
    threads.append(ActorCritic(env.copy(),
                               model=ActorCriticConv(),
                               optimizer=agent.opt,
                               trainable_weights=agent.weights,
                               name='Thread%s' % i))

trainer = AsyncTrainer(agent,
                       threads,
                       maxsteps=80000000,
                       batch_size=5,
                       logdir='/tmp/rf/A3C/%s' % env_name,
                       logfreq=240,
                       test_env=test_env,
                       test_render=True
                       )
trainer.train()
