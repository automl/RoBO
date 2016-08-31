
import numpy as np

import itertools


def init_grid(X_lower, X_upper, N):
    """
    Returns as initial design a grid with N samples

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
    n_dims = X_lower.shape[0]
    if np.power(N, n_dims) > 81 or n_dims > 4:
        raise AssertionError("Too many initial samples for grid")
    return np.array(itertools.product(
       *[np.linspace(X_lower[i], X_upper[i], N) for i in range(n_dims)]))
