from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
import sys

import numpy as np
import tensorflow as tf

from reinforceflow.version import version as __version__

__RANDOM_SEED__ = None


def set_random_seed(seed):
    if not isinstance(seed, int):
        raise ValueError('Random seed must be an integer value')
    global __RANDOM_SEED__
    __RANDOM_SEED__ = seed
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    from gym import spaces
    spaces.prng.seed(seed)


def get_random_seed():
    global __RANDOM_SEED__
    return __RANDOM_SEED__


handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s %(filename)s %(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger('.'.join(__name__.split('.')))
logger.propagate = False


def logger_setup():
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def undo_logger_setup():
    logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)


logger_setup()

del absolute_import
del division
del print_function
del sys
del logging
