
import numpy as np


def init_grid(lower, upper, n_points):
    """
    Returns as initial design a grid where each dimension is split into N intervals

    Parameters
    ----------
    lower: np.ndarray (D)
        Lower bounds of the input space
    upper: np.ndarray (D)
        Upper bounds of the input space
    n_points: int
        The number of points in each dimension

    Returns
    -------
    np.ndarray(N**lower.shape[0], D)
        The initial design data points
    """

    X = np.zeros([n_points ** lower.shape[0], lower.shape[0]])
    intervals = [np.linspace(lower[i], upper[i], n_points) for i in range(lower.shape[0])]
    m = np.meshgrid(*intervals)
    for i in range(lower.shape[0]):
        X[:, i] = m[i].flatten()

    return X
