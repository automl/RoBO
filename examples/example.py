'''
Created on Jun 15, 2015

@author: Aaron Klein
'''

import GPy
import matplotlib.pyplot as plt
import numpy as np

from robo.models.GPyModel import GPyModel
from robo.acquisition.EI import EI
from robo.maximizers.grid_search import GridSearch
from robo.recommendation.incumbent import compute_incumbent
from robo.task.base_task import BaseTask
from robo.visualization.plotting import plot_objective_function, plot_model,\
    plot_acquisition_function


# The optimization function that we want to optimize. It gets a numpy array with shape (N,D) where N >= 1 are the number of datapoints and D are the number of features
class ExampleTask(BaseTask):
    def __init__(self):
        self.X_lower = np.array([0])
        self.X_upper = np.array([7])
        self.n_dims = 1

    def objective_function(self, x):
        return np.sin(3 * x) * 4 * (x - 1) * (x + 2)

task = ExampleTask()

# Defining the method to model the objective function
kernel = GPy.kern.Matern52(input_dim=task.n_dims)
model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

# The acquisition function that we optimize in order to pick a new x
acquisition_func = EI(model, X_upper=task.X_upper, X_lower=task.X_lower, compute_incumbent=compute_incumbent, par=0.1)  # par is the minimum improvement that a point has to obtain


# Set the method that we will use to optimize the acquisition function
maximizer = GridSearch(acquisition_func, task.X_lower, task.X_upper)


# Draw one random point and evaluate it to initialize BO
X = np.array([np.random.uniform(task.X_lower, task.X_upper, task.n_dims)])
Y = task.objective_function(X)

# This is the main Bayesian optimization loop
for i in xrange(10):
    # Fit the model on the data we observed so far
    model.train(X, Y)

    # Update the acquisition function model with the retrained model
    acquisition_func.update(model)

    # Optimize the acquisition function to obtain a new point
    new_x = maximizer.maximize()

    # Evaluate the point and add the new observation to our set of previous seen points
    new_y = task.objective_function(np.array(new_x))
    X = np.append(X, new_x, axis=0)
    Y = np.append(Y, new_y, axis=0)

    # Visualize the objective function, model and the acquisition function
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1 = plot_objective_function(task.objective_function, task.X_lower, task.X_upper, X, Y, ax1)
    ax1 = plot_model(model, task.X_lower, task.X_upper, ax1)
    ax2 = plot_acquisition_function(acquisition_func, task.X_lower, task.X_upper, ax2)
    plt.show(block=True)
