# encoding=utf8
__author__ = "Lukas Voegtle"
__email__ = "voegtlel@tf.uni-freiburg.de"

import numpy as np


def init_random_normal(X_lower, X_upper, N, mean=None, std=None, seed=42):
    """
    Returns as initial design N data points sampled from a normal
    distribution.

    Parameters
    ----------
    X_lower: np.ndarray (D)
        Lower bounds of the input space
    X_upper: np.ndarray (D)
        Upper bounds of the input space
    N: int
        The number of initial data points
    mean: float
        Mean of the normal distribution
    std: float
        Std of the normal distribution
    seed: int
    	Number that is passed to the numpy random number generator

    Returns
    -------
    np.ndarray(N,D)
        The initial design data points
    """
    rng = np.random.RandomState(seed)
    if mean is None:
        mean = (X_upper + X_lower) * 0.5
    if std is None:
        std = (X_lower - X_upper) * 0.5
    n_dims = X_lower.shape[0]
    return np.array([np.clip(rng.normal(mean[i], std[i], N), X_lower[i], X_upper[i])
                     for i in range(n_dims)]).T
