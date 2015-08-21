'''
Created on Jun 12, 2015

@author: Aaron Klein
'''


import numpy as np


def plot_model(model, X_lower, X_upper, ax, resolution=0.1):
    X = np.arange(X_lower[0], X_upper[0], resolution)

    mean = np.zeros([X.shape[0]])
    var = np.zeros([X.shape[0]])
    for i in xrange(X.shape[0]):
        mean[i], var[i] = model.predict(X[i, np.newaxis, np.newaxis])

    ax.plot(X, mean, "b", label="Model")
    ax.fill_between(X, mean + 3 * np.sqrt(var), mean - 3 * np.sqrt(var), facecolor='blue', alpha=0.2)
    return ax


def plot_model_2d(model, X_lower, X_upper, ax, resolution=0.1):
    X1 = np.arange(X_lower[0], X_upper[0], resolution)
    X2 = np.arange(X_lower[1], X_upper[1], resolution)

    mean = np.zeros([X1.shape[0], X2.shape[0]])

    for i in xrange(X1.shape[0]):
        for j in xrange(X2.shape[0]):
            input = np.array([X1[i], X2[j]])
            input = input[np.newaxis, :]
            mean[i, j], v  = model.predict(input)
    #ax.axis([0, 6, 0, 6])
    ax.pcolormesh(mean)

    return ax


def plot_projected_model(model, X_lower, X_upper, ax, projection, resolution=0.1):
    X = np.arange(X_lower[0], X_upper[0], resolution)
    X = np.vstack((X, np.ones([X.shape[0]]) * projection))

    mean = np.zeros([X.shape[1]])
    var = np.zeros([X.shape[1]])
    for i in xrange(X.shape[1]):
        mean[i], var[i] = model.predict(np.array([X[:, i]]))

    ax.plot(X[0, :], mean, "b", label="Model")
    ax.fill_between(X[0, :], mean + 3 * np.sqrt(var), mean - 3 * np.sqrt(var), facecolor='blue', alpha=0.2)
    return ax


def plot_objective_function(objective_function, X_lower, X_upper, X, Y, ax, resolution=0.1):
    grid = np.arange(X_lower[0], X_upper[0], resolution)
    grid_values = objective_function(grid[:, np.newaxis])

    ax.plot(grid, grid_values[:, 0], "r", label="ObjectiveFunction")
    ax.plot(X, Y, "ro")
    return ax


def plot_objective_function_2d(objective_function, X_lower, X_upper, ax, resolution=0.1):
    grid1 = np.arange(X_lower[0], X_upper[0], resolution)
    grid2 = np.arange(X_lower[1], X_upper[1], resolution)

    a = np.tile(grid1, grid2.shape[0])
    b = np.repeat(grid2, grid1.shape[0])
    input = np.concatenate((a[:, np.newaxis], b[:, np.newaxis]), axis=1)
    grid_values = objective_function(input)

    ax.pcolormesh(grid_values)

    return ax


def plot_acquisition_function_2d(acq_fkt, X_lower, X_upper, ax, resolution=0.1):
    X1 = np.arange(X_lower[0], X_upper[0], resolution)
    X2 = np.arange(X_lower[1], X_upper[1], resolution)

    val = np.zeros([X1.shape[0], X2.shape[0]])

    for i in xrange(X1.shape[0]):
        for j in xrange(X2.shape[0]):
            input = np.array([X1[i], X2[j]])
            input = input[np.newaxis, :]
            val[i, j]= acq_fkt(input)
    #ax.axis([0, 6, 0, 6])
    ax.pcolormesh(val)

    return ax


def plot_acquisition_function(acquisition_function, X_lower, X_upper, ax, resolution=0.1, color="g", label="AcquisitionFunction"):
    grid = np.arange(X_lower[0], X_upper[0], resolution)

    grid_values = np.zeros([grid.shape[0]])
    for i in xrange(grid.shape[0]):
        grid_values[i] = acquisition_function(grid[i, np.newaxis])

    ax.plot(grid, grid_values, color, label=label)
    return ax
