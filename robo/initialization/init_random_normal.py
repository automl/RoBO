#encoding=utf8
__author__ = "Lukas Voegtle"
__email__ = "voegtlel@tf.uni-freiburg.de"

import numpy as np


def init_random_normal(x_lower, x_upper, n, mean=None, std=None):
    """
    Initializes by random normal distributed samples

    :param x_lower: lower bounds
    :param x_upper: upper bounds
    :param n: number of samples
    :param mean: mean for each dimension (n array). If None, the mean of the range is used.
    :param std: standard deviation for each dimension (n array). If None, the width of the range is used.
    :return: drawn samples
    """
    if mean == None:
        mean = (x_upper + x_lower) * 0.5
    if std == None:
        std = (x_lower - x_upper) * 0.5
    n_dims = x_lower.shape[0]
    return np.array([np.clip(np.random.normal(mean[i], std[i], n), x_lower[i], x_upper[i]) for i in range(n_dims)]).T