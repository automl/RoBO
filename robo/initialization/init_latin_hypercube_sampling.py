# encoding=utf8
__author__ = "Lukas Voegtle"
__email__ = "voegtlel@tf.uni-freiburg.de"

import numpy as np


def init_latin_hypercube_sampling(x_lower, x_upper, n):
    """
    Initializes by latin hypercube sampling (lhs)

    :param x_lower: lower bounds
    :param x_upper: upper bounds
    :param n: number of samples
    :return: drawn samples
    """
    n_dims = x_lower.shape[0]
    # Generate bounds for random number generator
    s_bounds = np.array([np.linspace(x_lower[i], x_upper[i], n + 1) for i in range(n_dims)])
    s_lower = s_bounds[:, :-1]
    s_upper = s_bounds[:, 1:]
    # Generate samples
    samples = s_lower + np.random.uniform(0, 1, s_lower.shape) * (s_upper - s_lower)
    # Shuffle samples in each dimension
    for i in range(n_dims):
        np.random.shuffle(samples[i, :])
    return samples.T
