"""
Defines objective 2 as the Hartmann 6 function:
z(X) = - \sum_{i=1}^{4} c_i exp(-\sum_{j=1}^{6} alpha_{ij} * (x_{ij} - p_{ij})**2)
obj_2(Z, S) = z(s_1, s_2, z_1, s_3, z_2, s_4)
"""

import numpy as np

import scipy
import scipy.optimize


_hartman6_alpha = [[10, 3, 17, 3.5, 1.7, 8],
                   [0.05, 10, 17, .1, 8, 14],
                   [3, 3.5, 1.7, 10, 17, 8],
                   [17, 8, 0.05, 10, 0.1, 14]]
_hartman6_c = [1, 1.2, 3, 3.2]
_hartman6_p = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
               [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
               [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
               [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]

def hartman6(x):
    sum1 = 0
    for i in range(4):
        sum2 = 0
        for j in range(6):
            sum2 -= _hartman6_alpha[i][j] * (x[j] - _hartman6_p[i][j]) ** 2
        sum1 -= _hartman6_c[i] * np.exp(sum2)
    return np.array(sum1, ndmin=2)

def hartman6_dx(x, ix):
    """
    Gets the derivative for one dimension ix
    :param x: position
    :param ix: index of x for which to get the derivative
    :return: derivative
    """
    sum1 = 0
    for i in range(4):
        sum2 = 0
        for j in range(6):
            sum2 -= _hartman6_alpha[i][j] * (x[j] - _hartman6_p[i][j]) ** 2
        sum1 += _hartman6_c[i] * np.exp(sum2) * 2 * _hartman6_alpha[i][ix] * (x[ix] - _hartman6_p[i][ix])
    return sum1

_objective2_min_start = [[0.1312, 0.1696, 0.0124, 0.5886],
                         [0.2329, 0.4135, 0.3736, 0.9991],
                         [0.2348, 0.1451, 0.2883, 0.6650],
                         [0.4047, 0.8828, 0.5743, 0.0381]]

def objective2(Z, S):
    """
    Second objective function
    :param Z: context [0, 1]^2
    :param S: action [0, 1]^4
    :return: value for the point(s)
    """
    return hartman6(np.concatenate((S[:, 0:2], Z[:, 0:1], S[:, 2:3], Z[:, 1:2], S[:, 3:4]), axis=1).flatten())

def objective2_grad(Z, S):
    """
    Gradient of the second objective function for a fixed context
    :param Z: context [0, 1]^2
    :param S: action [0, 1]^4
    :return: value for the point(s)
    """
    X = np.concatenate((S[0:2], Z[0:1], S[2:3], Z[1:2], S[3:4]))
    return np.array((hartman6_dx(X, 0), hartman6_dx(X, 1), hartman6_dx(X, 3), hartman6_dx(X, 5)))

def objective2_min_action(Z):
    """
    Calculates the location (action) of the minimum for a given context
    :param Z: context
    :return: locations of minimums
    """
    S = np.zeros((Z.shape[0], 4))
    for i in range(Z.shape[0]):
        z = Z[i, :].flatten()

        def minfn(s):
            return objective2(z[np.newaxis, :], s[np.newaxis, :]).flatten(), objective2_grad(z, s)
        res = np.random.uniform(size=4)
        fun = np.Inf
        for x0 in _objective2_min_start:
            resi = scipy.optimize.minimize(fun=minfn, x0=x0, method='L-BFGS-B', jac=True, bounds=((0,1),)*4)
            if resi.fun[0] < fun:
                res = resi.x
                fun = resi.fun[0]
        # This won't improve the result significantly
        #for _ in range(1000):
        #    resi = scipy.optimize.minimize(fun=minfn, x0=np.random.uniform(size=4), jac=True, bounds=((0,1),)*4)
        #    if resi.fun[0] < fun:
        #        print "found better: ", resi.fun[0], " < ", fun
        #        res = resi.x
        #        fun = resi.fun[0]
        S[i, :] = res
    return S

def objective2_min(Z):
    """
    Calculates the location (action) of the minimum for a given context
    :param Z: context
    :return: locations of minimums
    """
    return objective2(Z, objective2_min_action(Z))

objective2_S_lower = np.array([0, 0, 0, 0])
objective2_S_upper = np.array([1, 1, 1, 1])
objective2_dims_Z = 2
objective2_dims_S = 4