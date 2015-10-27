#encoding=utf8
__author__ = "Lukas Voegtle"
__email__ = "voegtlel@tf.uni-freiburg.de"

import numpy as np


def init_random_normal(x_lower, x_upper, mean, std, n):
    """
    Initializes by random normal distributed samples

    :param x_lower: lower bounds
    :param x_upper: upper bounds
    :param mean: mean for each dimension (n array)
    :param std: standard deviation for each dimension (n array)
    :param n: number of samples
    :return: drawn samples
    """
    n_dims = x_lower.shape[0]
    return np.transpose(
        np.array([np.clip(np.random.normal(mean[i], std[i], n), x_lower[i], x_upper[i]) for i in range(n_dims)])
    )