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
from reinforceflow.agents import DeepQ
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.models import DeepQModel
from reinforceflow.core.optimizer import RMSProp
from reinforceflow.envs import AtariWrapper
from reinforceflow.trainers.async_trainer import AsyncTrainer

reinforceflow.set_random_seed(555)

env_name = "BreakoutNoFrameskip-v4"
env = AtariWrapper(env_name,
                   action_repeat=4,
                   obs_stack=4,
                   new_width=84,
                   new_height=84,
                   to_gray=True,
                   noop_action=[1, 0, 0, 0],
                   start_action=[0, 1, 0, 0],
                   clip_rewards=True)

test_env = AtariWrapper(env_name,
                        action_repeat=4,
                        obs_stack=4,
                        new_width=84,
                        new_height=84,
                        to_gray=True,
                        start_action=[0, 1, 0, 0])

agent = DeepQ(env,
              use_double=False,
              model=DeepQModel(nature_arch=False, dueling=False),
              optimizer=RMSProp(7e-4, decay=0.99, epsilon=0.1))
threads = []
for i, eps in enumerate([0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.5, 0.5] * 2):
    threads.append(DeepQ(env.copy(),
                         model=DeepQModel(nature_arch=False, dueling=False),
                         optimizer=agent.opt,
                         trainable_weights=agent.weights,
                         target_net=agent.target_net,
                         target_weights=agent.target_weights,
                         use_double=False,
                         targetfreq=10000,
                         policy=EGreedyPolicy(1.0, eps, 4000000),
                         name='thread%s' % i))

trainer = AsyncTrainer(agent,
                       threads,
                       maxsteps=80000000,
                       batch_size=5,
                       logdir='/tmp/rf/AsyncDeepQ/%s' % env_name,
                       logfreq=240,
                       test_env=test_env
                       )
trainer.train()
