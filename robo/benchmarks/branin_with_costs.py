'''
Created on Jun 29, 2015

@author: Aaron Klein
'''

import numpy as np

from robo.benchmarks.branin import branin, get_branin_bounds
from time import sleep


def branin_with_costs(x):
    print x
    y = branin(x[:, :-1])
    s = x[0, -1]
    sleep(s)
    return y


def get_branin_with_costs_bounds():
    X_lower, X_upper, n_dims = get_branin_bounds()
    X_lower = np.concatenate((X_lower, np.array([1])))
    X_upper = np.concatenate((X_upper, np.array([1000])))
    n_dims += 1
    is_env = np.array([0, 0, 1])
    return X_lower, X_upper, n_dims, is_env

