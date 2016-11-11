
import numpy as np

import itertools


def init_grid(X_lower, X_upper, N):
    """
    Returns as initial design a grid where each dimension is split into N intervals

    Parameters
    ----------
    X_lower: np.ndarray (D)
        Lower bounds of the input space
    X_upper: np.ndarray (D)
        Upper bounds of the input space
    N: int
        The number of points in each dimension

    Returns
    -------
    np.ndarray(N**X_lower.shape[0], D)
        The initial design data points
    """

    X = np.zeros([N ** X_lower.shape[0], X_lower.shape[0]])
    intervals = [np.linspace(X_lower[i], X_upper[i], N) for i in range(X_lower.shape[0])]
    m = np.meshgrid(*intervals)
    for i in range(X_lower.shape[0]):
        X[:, i] = m[i].flatten()

    return X