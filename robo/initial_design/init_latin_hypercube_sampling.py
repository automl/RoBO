
import numpy as np


def init_latin_hypercube_sampling(lower, upper, n_points, rng=None):
    """
    Returns as initial design a N data points sampled from a latin hypercube.

    Parameters
    ----------
    lower: np.ndarray (D)
        Lower bound of the input space
    upper: np.ndarray (D)
        Upper bound of the input space
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
    # Generate bounds for random number generator
    s_bounds = np.array([np.linspace(lower[i], upper[i], n_points + 1) for i in range(n_dims)])
    s_lower = s_bounds[:, :-1]
    s_upper = s_bounds[:, 1:]
    # Generate samples
    samples = s_lower + rng.uniform(0, 1, s_lower.shape) * (s_upper - s_lower)
    # Shuffle samples in each dimension
    for i in range(n_dims):
        rng.shuffle(samples[i, :])
    return samples.T
