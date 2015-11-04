#encoding=utf8
__author__ = "Lukas Voegtle"
__email__ = "voegtlel@tf.uni-freiburg.de"

import numpy as np


def init_random_uniform(x_lower, x_upper, n):
    """
    Initializes by random uniform samples

    :param x_lower: lower bounds
    :param x_upper: upper bounds
    :param n: number of samples
    :return: drawn samples
    """
    n_dims = x_lower.shape[0]
    return np.array([np.random.uniform(x_lower, x_upper, n_dims) for _ in range(n)])