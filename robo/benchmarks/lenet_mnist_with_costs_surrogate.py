'''
Created on Jul 1, 2015

@author: Aaron Klein
'''


import numpy as np

from sklearn.ensemble import RandomForestRegressor
from ParameterConfigSpace.config_space import ConfigSpace


def lenet_mnist_with_cost_surrogate(x):
    X = np.load("robo/benchmarks/data_lenet_mnist_with_cost_surrogate/data.npy")
    y = np.load("robo/benchmarks/data_lenet_mnist_with_cost_surrogate/targets.npy")

    rf = RandomForestRegressor()
    rf.fit(X, y)
    y_hat = np.array([rf.predict(x)])
    return y_hat


def get_lenet_mnist_with_cost_surrogate_bounds():
    #cs = ConfigSpace("robo/benchmarks/data_lenet_mnist_with_cost_surrogate/param.pcs")
    #n_dims = len(cs.parameters)
    #X_lower = np.zeros([n_dims])
    #X_upper = np.zeros([n_dims])
    X = np.load("robo/benchmarks/data_lenet_mnist_with_cost_surrogate/data.npy")
    n_dims = 8
    X_lower = np.min(X, axis=0)
    X_upper = np.max(X, axis=0)
    is_env = np.zeros([n_dims])
    #for i, name in enumerate(cs.get_parameter_names()):
    #    if 'env' in name:
    #        is_env[i] = 1
    #    X_lower[i], X_upper[i] = cs.parameters[name].values

    return X_lower, X_upper, n_dims, is_env
