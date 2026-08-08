import random
import numpy as np
import tensorflow as tf


def set_seed(seed=42):
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)