#encoding=utf8
__author__ = "Lukas Voegtle"
__email__ = "voegtlel@tf.uni-freiburg.de"

import numpy as np

import itertools

def init_grid(x_lower, x_upper, n):
    """
    Initializes with grid samples

    :param x_lower: lower bounds
    :param x_upper: upper bounds
    :param n: number of samples
    :return: drawn samples
    """
    n_dims = x_lower.shape[0]
    if np.power(n, n_dims) > 81 or n_dims > 4:
        raise AssertionError("Too many initial samples for grid")
    return np.array(itertools.product(*[np.linspace(x_lower[i], x_upper[i], n) for i in range(n_dims)]))