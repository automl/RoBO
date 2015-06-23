'''
Created on Jun 23, 2015

@author: Aaron Klein
'''

import numpy as np


def hartmann6(params):
    """6d Hartmann test function
        input bounds:  0 <= xi <= 1, i = 1..6
        global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
        min function value = -3.32236
    """
    print params

    x = params[0, 0]
    y = params[0, 1]
    z = params[0, 2]
    a = params[0, 3]
    b = params[0, 4]
    c = params[0, 5]

    value = np.array([x, y, z, a, b, c])

    a = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
               [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
               [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
               [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])
    c = np.array([1.0, 1.2, 3.0, 3.2])
    p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
               [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
               [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
               [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    s = 0
    for i in [0, 1, 2, 3]:
        sm = a[i, 0] * (value[0] - p[i, 0]) ** 2
        sm += a[i, 1] * (value[1] - p[i, 1]) ** 2
        sm += a[i, 2] * (value[2] - p[i, 2]) ** 2
        sm += a[i, 3] * (value[3] - p[i, 3]) ** 2
        sm += a[i, 4] * (value[4] - p[i, 4]) ** 2
        sm += a[i, 5] * (value[5] - p[i, 5]) ** 2
        s += c[i] * np.exp(-sm)
    result = -s
    return np.array([[result]])


def get_hartmann6_bounds():
    X_lower = np.array([0, 0, 0, 0, 0, 0])
    X_upper = np.array([1, 1, 1, 1, 1, 1])
    n_dims = 6
    return X_lower, X_upper, n_dims
