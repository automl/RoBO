import GPy
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import numpy as np
import random

from robo.models.GPyModel import GPyModel
from robo.acquisition.EI import EI
from robo.maximizers.maximize import stochastic_local_search


def objective_function(x):
    return  np.sin(3 * x) * 4 * (x - 1) * (x + 2)

# The bounds and dimensions of our configuration space
X_lower = np.array([0])
X_upper = np.array([6])
dims = 1

# Specify the maximizer for the acquisition function
maximizer = stochastic_local_search

# Setup the model and its parameters
kernel = GPy.kern.Matern52(input_dim=dims)
model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

# Specify the acquisition function
acquisition_func = EI(model, X_upper=X_upper, X_lower=X_lower, par=0.1)  # par is the minimum improvement

# Randomly sample a point in order to fit the model
X = np.empty((1, dims))
for i in xrange(dims):
    X[0, i] = random.random() * (X_upper[i] - X_lower[i]) + X_lower[i];
    x = np.array(X)

# Evaluate the objective function at that point
Y = objective_function(X)

# Main Bayesian optimization loop
for i in xrange(10):
    # Train the model at the configuration, performance pairs
    model.train(X, Y)

    # Update the acquisition function with the new trained model
    acquisition_func.update(model)

    # Maximize the acquisition function in order to get a new configuration
    new_x = maximizer(acquisition_func, X_lower, X_upper)

    # Evaluate the new configuration
    new_y = objective_function(np.array(new_x))

    # Add the new configuration and its performance to our observed data points
    X = np.append(X, new_x, axis=0)
    Y = np.append(Y, new_y, axis=0)

    # Plotting
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    plotting_range = np.linspace(X_lower[0], X_upper[0], num=1000)
    ax1.plot(plotting_range, objective_function(plotting_range[:, np.newaxis]), color='b', linestyle="--")
    _min_y1, _max_y1 = ax1.get_ylim()
    model.visualize(ax1, X_lower[0], X_upper[0])
    _min_y2, _max_y2 = ax1.get_ylim()
    ax1.set_ylim(min(_min_y1, _min_y2), max(_max_y1, _max_y2))
    mu, var = model.predict(new_x)
    ax1.plot(new_x[0], mu[0], "r.", markeredgewidth=5.0)
    ax2 = acquisition_func.plot(fig, X_lower[0], X_upper[0], plot_attr={"color":"red"}, resolution=1000)

    plt.show(block=True)
