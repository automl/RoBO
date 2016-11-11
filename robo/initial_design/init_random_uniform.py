
import numpy as np


def init_random_uniform(lower, upper, n_points, rng=None):
    """
    Samples N data points uniformly.

    Parameters
    ----------
    lower: np.ndarray (D)
        Lower bounds of the input space
    upper: np.ndarray (D)
        Upper bounds of the input space
    n_points: int
        The number of initial data points
    rng: np.random.RandomState
            Random number generator
    Returns
    -------
    np.ndarray(N,D)
        The initial design data points
    """

    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    n_dims = lower.shape[0]

    return np.array([rng.uniform(lower, upper, n_dims) for _ in range(n_points)])
