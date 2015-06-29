'''
Created on Jun 26, 2015

@author: Aaron Klein
'''


import numpy as np


def goldstein_price(x):
    fval = np.array(1 + (x[:, 0] + x[:, 1] + 1) ** 2 * (19 - 14 * x[:, 0] + 3 * x[:, 0] ** 2 -
                                                 14 * x[:, 1] + 6 * x[:, 0] * x[:, 1] + 3 * x[:, 1] ** 2)) * (30 + (2 * x[:, 0] - 3 * x[:, 1]) ** 2 * (18 - 32 * x[:, 0] +
                                                                                             12 * x[:, 0] ** 2 + 48 * x[:, 1] - 36 * x[:, 0] * x[:, 1] + 27 * x[:, 1] ** 2))
    return fval


def get_goldstein_price_bounds():
    X_lower = np.array([-2, -2])
    X_upper = np.array([2, 2])
    n_dims = 2
    return X_lower, X_upper, n_dims
