"""
Defines objective 1 as the product of two branin functions:
y_b = (v - 5.1/(4*pi^2)*u**2 + 5/pi*u-6)^2 + 10*(1-1/(8*pi))*cos(u) + 10
obj_1(Z, S) = y_b(15 * z_1 - 5, 15 * s_1) * y_b(15 * s_2 - 5, 15 * z_2)
"""

import numpy as np

import scipy
import scipy.optimize

_branin_k1 = 5.1/(4*np.pi*np.pi)
_branin_k2 = 5 / np.pi
_branin_k3 = 10 * (1-1/(8*np.pi))

def branin(u, v):
    """
    The branin function.
    :param u: limits: [-5, 10]
    :param v: limits: [0, 15]
    :return:
    """
    w = (v - _branin_k1 * u * u + _branin_k2 * u - 6)
    return w * w + _branin_k3 * np.cos(u) + 10


def branin_grad_u(u, v):
    """
    Gradient of the branin function.
    :param u: limits: [-5, 10]
    :param v: limits: [0, 15]
    :return:
    """
    return 2*(_branin_k2 - 2*_branin_k1*u)*(-_branin_k1*u*u+_branin_k2*u+v-6)-_branin_k2*np.sin(u)


def branin_grad_v(u, v):
    """
    Gradient of the branin function.
    :param u: limits: [-5, 10]
    :param v: limits: [0, 15]
    :return:
    """
    return 2*(-_branin_k1*u*u + _branin_k2*u + v - 6)


def branin_min_u(u):
    """
    Returns the position of the minimum for a fixed u
    :param u:
    :return: v
    """
    return _branin_k1 * u * u - _branin_k2 * u + 6

def branin_min_v(v):
    """
    Returns the position of the minimum for a fixed v
    :param v:
    :return: u
    """
    res = np.zeros(v.shape)
    for i, v_ in enumerate(v):
        def minfn(x):
            return branin(x, v_), branin_grad_u(x, v_)
        res[i, :] = scipy.optimize.minimize(fun=minfn, x0=np.array((0.5,)), method='L-BFGS-B', jac=True, bounds=((-5,10),)).x[0]
    return res

def objective1(Z, S):
    """
    This is a 4 dimensional function based on two branin evaluations
    :param Z: context [0, 1]^2
    :param S: action [0, 1]^2
    :return: value for the point(s)
    """
    x2, x3 = S[:, 0:1], S[:, 1:2]
    x1, x4 = Z[:, 0:1], Z[:, 1:2]
    return branin(15*x1 - 5, 15*x2) * branin(15*x3 - 5, 15*x4)

def objective1_min_action(Z):
    """
    Calculates the location (action) of the minimum for a given context
    :param Z: context
    :return: location of minimum as tuple
    """
    x1, x4 = Z[:, 0:1], Z[:, 1:2]
    x2 = branin_min_u(15*x1 - 5) / 15
    x3 = (branin_min_v(15*x4) + 5) / 15
    return np.concatenate((x2, x3), axis=1)

def objective1_min(Z):
    """
    Calculates the minimum for a given context
    :param Z: context
    :return: value of the minimum
    """
    x1, x4 = Z[:, 0:1], Z[:, 1:2]
    u1 = 15*x1 - 5
    v2 = 15*x4
    return branin(u1, branin_min_u(u1)) * branin(branin_min_v(v2), v2)

objective1_S_lower = np.array([0, 0])
objective1_S_upper = np.array([1, 1])
objective1_dims_Z = 2
objective1_dims_S = 2
