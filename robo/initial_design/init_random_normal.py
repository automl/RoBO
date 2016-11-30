
import numpy as np


def init_random_normal(lower, upper, n_points, mean=None, std=None, rng=None):
    """
    Returns as initial design N data points sampled from a normal
    distribution.

    Parameters
    ----------
    lower: np.ndarray (D)
        Lower bound of the input space
    upper: np.ndarray (D)
        Upper bound of the input space
    n_points: int
        The number of initial data points
    mean: np.ndarray (D)
        Mean of the normal distribution for each dimension
    std: np.ndarray (D)
        Std of the normal distribution for each dimension
    rng: np.random.RandomState
            Random number generator

    Returns
    -------
    np.ndarray(N, D)
        The initial design data points
    """
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    n_dims = lower.shape[0]
    # take the center as mean
    if mean is None:
        mean = (upper + lower) * 0.5

    if std is None:
        std = np.ones([n_dims]) * 0.1

    return np.array([np.clip(rng.normal(mean[i], std[i], n_points), lower[i], upper[i])
                     for i in range(n_dims)]).T
