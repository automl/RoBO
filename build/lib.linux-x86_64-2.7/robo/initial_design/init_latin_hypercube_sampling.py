# encoding=utf8
__author__ = "Lukas Voegtle"
__email__ = "voegtlel@tf.uni-freiburg.de"

import numpy as np


def init_latin_hypercube_sampling(X_lower, X_upper, N, rng=None):
    """
    Returns as initial design a N data points sampled from a latin hypercube.

    Parameters
    ----------
    X_lower: np.ndarray (D)
        Lower bounds of the input space
    X_upper: np.ndarray (D)
        Upper bounds of the input space
    N: int
        The number of initial data points

    Returns
    -------
    np.ndarray(N,D)
        The initial design data points
    """
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))
    n_dims = X_lower.shape[0]
    # Generate bounds for random number generator
    s_bounds = np.array([np.linspace(X_lower[i], X_upper[i], N + 1) for i in range(n_dims)])
    s_lower = s_bounds[:, :-1]
    s_upper = s_bounds[:, 1:]
    # Generate samples
    samples = s_lower + rng.uniform(0, 1, s_lower.shape) * (s_upper - s_lower)
    # Shuffle samples in each dimension
    for i in range(n_dims):
        rng.shuffle(samples[i, :])
    return samples.T
