'''
Created on Jun 12, 2015

@author: Aaron Klein
'''


import numpy as np


def plot_model(model, X_lower, X_upper, ax, resolution=0.1):
    X = np.arange(X_lower[0], X_upper[0], resolution)

    mean = np.zeros([X.shape[0]])
    std = np.zeros([X.shape[0]])
    for i in xrange(X.shape[0]):
        mean[i], std[i] = model.predict(X[i, np.newaxis, np.newaxis])

    ax.plot(X, mean, "b", label="Model")
    ax.fill_between(X, mean +  std, mean - std, facecolor='blue', alpha=0.2)
    return ax


def plot_objective_function(objective_function, X_lower, X_upper, X, Y, ax, resolution=0.1):
    grid = np.arange(X_lower[0], X_upper[0], resolution)

    grid_values = np.zeros([grid.shape[0]])
    for i in xrange(grid.shape[0]):
        grid_values[i] = objective_function(grid[i])

    ax.plot(grid, grid_values, "r", label="ObjectiveFunction")
    ax.plot(X, Y, "ro")
    return ax


def plot_acquisition_function(acquisition_function, X_lower, X_upper, ax, resolution=0.1):
    grid = np.arange(X_lower[0], X_upper[0], resolution)

    grid_values = np.zeros([grid.shape[0]])
    for i in xrange(grid.shape[0]):
        grid_values[i] = acquisition_function(grid[i, np.newaxis])

    ax.plot(grid, grid_values, "g", label="AcquisitionFunction")
    #ax.plot(X, y, "ro")
    return ax
