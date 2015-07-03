'''
Created on Jul 1, 2015

@author: Aaron Klein
'''


import numpy as np

from sklearn.ensemble import RandomForestRegressor
from ParameterConfigSpace.config_space import ConfigSpace


def lenet_mnist_with_costs_surrogate(x):
    X = np.load("robo/benchmarks/data_lenet_mnist_with_costs_surrogate/data.npy")
    y = np.load("robo/benchmarks/data_lenet_mnist_with_costs_surrogate/targets.npy")

    rf = RandomForestRegressor()
    rf.fit(X, y)
    y_hat = np.array([rf.predict(x)])
    return y_hat


def get_lenet_mnist_surrogate_bounds():
    cs = ConfigSpace("robo/benchmarks/data_lenet_mnist_with_costs_surrogate/param.pcs")
    n_dims = len(cs.parameters)
    X_lower = np.zeros([n_dims])
    X_upper = np.zeros([n_dims])
    for i, name in enumerate(cs.get_parameter_names()):
        X_lower[i], X_upper[i] = cs.parameters[name].values
    return X_lower, X_upper, n_dims
