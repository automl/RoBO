# encoding=utf8
__author__ = "Lukas Voegtle"
__email__ = "voegtlel@tf.uni-freiburg.de"

import numpy as np


def init_random_uniform(X_lower, X_upper, N, rng=None):
    """
    Samples N data points uniformly.

    Parameters
    ----------
    X_lower: np.ndarray (D)
        Lower bounds of the input space
    X_upper: np.ndarray (D)
        Upper bounds of the input space
    N: int
        The number of initial data points
    seed: int
        Number that is passed to the numpy random number generator

    Returns
    -------
    np.ndarray(N,D)
        The initial design data points
    """
    if rng is None:
        rng = np.random.RandomState(42)
    n_dims = X_lower.shape[0]
    return np.array([rng.uniform(X_lower, X_upper, n_dims) for _ in range(N)])
