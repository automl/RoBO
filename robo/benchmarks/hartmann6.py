'''
Created on Jun 23, 2015

@author: Aaron Klein
'''

import numpy as np


def hartmann6(x):
    """6d Hartmann test function
        input bounds:  0 <= xi <= 1, i = 1..6
        global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
        min function value = -3.32237
    """
    alpha = [1.00, 1.20, 3.00, 3.20]
    A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                  [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                  [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                  [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
    P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                           [2329, 4135, 8307, 3736, 1004, 9991],
                           [2348, 1451, 3522, 2883, 3047, 6650],
                           [4047, 8828, 8732, 5743, 1091, 381]])

    external_sum = np.zeros((x.shape[0], 1))
    for i in range(4):
        internal_sum = 0
        for j in range(6):
            internal_sum = internal_sum + A[i, j] * (x[:, j] - P[i, j]) ** 2
        external_sum = external_sum + alpha[i] * np.exp(-internal_sum)
    return np.array(-external_sum)


def get_hartmann6_bounds():
    X_lower = np.array([0, 0, 0, 0, 0, 0])
    X_upper = np.array([1, 1, 1, 1, 1, 1])
    n_dims = 6
    return X_lower, X_upper, n_dims
