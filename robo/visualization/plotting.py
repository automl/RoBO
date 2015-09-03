'''
Created on Jun 12, 2015

@author: Aaron Klein
Edits: Numair Mansur (numair.mansur@gmail.com)
'''


import numpy as np


def plot_model(model, X_lower, X_upper, ax, resolution=0.1,mean_color='r',uncertainity_color='blue',label="Model",std_scale=3,plot_mean=True):
    X = np.arange(X_lower[0], X_upper[0], resolution)

    mean = np.zeros([X.shape[0]])
    var = np.zeros([X.shape[0]])
    for i in xrange(X.shape[0]):
        mean[i], var[i] = model.predict(X[i, np.newaxis, np.newaxis])

    if plot_mean == True:
        ax.plot(X, mean, mean_color, label=label)
    ax.fill_between(X, mean + std_scale * np.sqrt(var), mean - std_scale * np.sqrt(var), facecolor=uncertainity_color, alpha=0.2)
    return ax


def plot_objective_function(objective_function, X_lower, X_upper, X, Y, ax, resolution=0.1,color='black',label='ObjectiveFunction',maximizer_flag=True):
    grid = np.arange(X_lower[0], X_upper[0], resolution)

    grid_values = np.zeros([grid.shape[0]])
    for i in xrange(grid.shape[0]):
        grid_values[i] = objective_function(grid[i])

    ax.plot(grid, grid_values, color, label=label,linestyle="--")
    if maximizer_flag ==True:
        ax.plot(X[0], Y[0], "bo")
        ax.plot(X[1],Y[1], "rv")
    return ax


def plot_acquisition_function(acquisition_function, X_lower, X_upper,X, ax, resolution=0.1,label="AcquisitionFunction", maximizer_flag = True):
    grid = np.arange(X_lower[0], X_upper[0], resolution)

    grid_values = np.zeros([grid.shape[0]])
    for i in xrange(grid.shape[0]):
        grid_values[i] = acquisition_function(grid[i, np.newaxis])

    ax.plot(grid, grid_values, "g", label=label)
    if maximizer_flag == True:
        ax.plot(X[1],np.amax(grid_values), "rv")
    return ax
