'''
Created on Jun 12, 2015

@author: Aaron Klein
Edits: Numair Mansur (numair.mansur@gmail.com)
'''


import numpy as np


def plot_model(
        model,
        X_lower,
        X_upper,
        ax,
        resolution=0.1,
        maximizer=None,
        mean_color='b',
        uncertainity_color='blue',
        label="Model",
        std_scale=3,
        plot_mean=True):
    ''' Plots the model on the ax object passed to it

    Args:
        model (object): Model of the objective funciton
        X_lower (np.array): Upper bound of the input space
        X_upper (np.array): Lower bound of the input space
        ax (object): subplot for the model and the objective funciton
        resolution (float): resolution for the subplot
        mean_color (string): Color of the prosterior mean
        uncertainity_color (string): Color of the model
        label (string): Label string
        std_scale (int): Standard Deviation Scale
        plot_mean (bool): Bool flag, Plot the mean curve if value is True

    Returns:
        ax (object) : subplot for the model and the objective funciton

    '''

    X = np.arange(X_lower[0], X_upper[0], resolution)

    mean = np.zeros([X.shape[0]])
    var = np.zeros([X.shape[0]])
    for i in xrange(X.shape[0]):
        mean[i], var[i] = model.predict(X[i, np.newaxis, np.newaxis])

    if plot_mean:
        ax.plot(X, mean, mean_color, label=label)
    if maximizer is not None:
        ax.axvline(maximizer, color='red')
    ax.fill_between(
        X,
        mean +
        std_scale *
        np.sqrt(var),
        mean -
        std_scale *
        np.sqrt(var),
        facecolor=uncertainity_color,
        alpha=0.2)
    ax.legend()
    return ax


def plot_objective_function(
        objective_function,
        X_lower,
        X_upper,
        ax,
        X=None,
        Y=None,
        resolution=0.1,
        color='black',
        label='ObjectiveFunction'):
    ''' Plots the objective_function on the ax object passed to it

    Args:
        objective_function ():
        X_lower ():
        X_upper ():
        X ():
        Y ():
        ax ():
        resolution ():
        color ():
        label ():
        maximizer_flag ():
    Returns:
        ax ():

    '''
    grid = np.arange(X_lower[0], X_upper[0], resolution)

    grid_values = np.zeros([grid.shape[0]])
    for i in xrange(grid.shape[0]):
        grid_values[i] = objective_function(grid[i])

    ax.plot(grid, grid_values, color, label=label, linestyle="--")
    if X is not None and Y is not None:
        ax.plot(X, Y, "bo")
    ax.legend()
    return ax


def plot_acquisition_function(
        acquisition_function,
        X_lower,
        X_upper,
        ax,
        resolution=0.1,
        label="AcquisitionFunction",
        maximizer=None):
    ''' Plots the acquisition_function on the ax object passed to it

    Args:
        acquisition_function ():
        X_lower ():
        X_upper ():
        X ():
        ax ():
        resolution ():
        label ():
        maximizer_flag ():
    Returns:
        ax ():

    '''
    grid = np.arange(X_lower[0], X_upper[0], resolution)

    grid_values = np.zeros([grid.shape[0]])
    for i in xrange(grid.shape[0]):
        grid_values[i] = acquisition_function(grid[i, np.newaxis])

    ax.plot(grid, grid_values, "g", label=label)
    if maximizer is not None:
        ax.axvline(maximizer, color='red')
    ax.legend()
    return ax
