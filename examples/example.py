'''
Created on Jun 15, 2015

@author: Aaron Klein
'''

import GPy
import matplotlib; matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import numpy as np

from robo.models.GPyModel import GPyModel
from robo.acquisition.EI import EI
from robo.maximizers.maximize import stochastic_local_search
from robo.recommendation.incumbent import compute_incumbent


# The optimization function that we want to optimize. It gets a numpy array with shape (N,D) where N >= 1 are the number of datapoints and D are the number of features
def objective_function(x):
    return  np.sin(3 * x) * 4 * (x - 1) * (x + 2)

# Defining the bounds and dimensions of the input space
X_lower = np.array([0])
X_upper = np.array([6])
dims = 1

# Set the method that we will use to optimize the acquisition function
maximizer = stochastic_local_search

# Defining the method to model the objective function
kernel = GPy.kern.Matern52(input_dim=dims)
model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

# The acquisition function that we optimize in order to pick a new x
acquisition_func = EI(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1)  # par is the minimum improvement that a point has to obtain

# Draw one random point and evaluate it to initialize BO
X = np.array([np.random.uniform(X_lower, X_upper, dims)])
Y = objective_function(X)

# This is the main Bayesian optimization loop
for i in xrange(10):
    # Fit the model on the data we observed so far
    model.train(X, Y)

    # Update the acquisition function model with the retrained model
    acquisition_func.update(model)

    # Optimize the acquisition function to obtain a new point 
    new_x = maximizer(acquisition_func, X_lower, X_upper)

    # Evaluate the point and add the new observation to our set of previous seen points
    new_y = objective_function(np.array(new_x))
    X = np.append(X, new_x, axis=0)
    Y = np.append(Y, new_y, axis=0)

    # Visualize the objective function, model and the acquisition function
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
    ax2 = acquisition_func.plot(fig, X_lower[0], X_upper[0], plot_attr={"color": "red"}, resolution=1000)

plt.show(block=True)
