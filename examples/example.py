'''
Created on Jun 15, 2015

@author: Aaron Klein
'''
import setup_logger

import GPy
import matplotlib.pyplot as plt
import numpy as np

from robo.models.gpy_model import GPyModel
from robo.acquisition.ei import EI
from robo.maximizers.grid_search import GridSearch
from robo.recommendation.incumbent import compute_incumbent
from robo.task.base_task import BaseTask
from robo.visualization.plotting import plot_objective_function, plot_model,\
    plot_acquisition_function
from robo.initial_design.init_random_uniform import init_random_uniform


# The optimization function that we want to optimize.
# It gets a numpy array with shape (N,D) where N >= 1 are the number of
# datapoints and D are the number of features
class ExampleTask(BaseTask):

    def __init__(self):
        X_lower = np.array([0])
        X_upper = np.array([7])
        super(ExampleTask, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        return np.sin(3 * x) * 4 * (x - 1) * (x + 2)

task = ExampleTask()

# Defining the method to model the objective function
kernel = GPy.kern.Matern52(input_dim=task.n_dims)
model = GPyModel(kernel, optimize=True, num_restarts=10)

# The acquisition function that we optimize in order to pick a new x
acquisition_func = EI(model, X_upper=task.X_upper, X_lower=task.X_lower,
                      compute_incumbent=compute_incumbent,
                      par=0.1)  # par is the minimum required improvement


# Set the method that we will use to optimize the acquisition function
maximizer = GridSearch(acquisition_func, task.X_lower, task.X_upper)


# Draw three random points and evaluate them to initialize BO
X = init_random_uniform(task.X_lower, task.X_upper, 3)
Y = task.evaluate(X)

# This is the main Bayesian optimization loop
for i in xrange(10):
    # Fit the model on the data we observed so far
    model.train(X, Y)

    # Update the acquisition function model with the retrained model
    acquisition_func.update(model)

    # Optimize the acquisition function to obtain a new point
    new_x = maximizer.maximize()

    # Evaluate the point and add the new observation to our set of observations
    new_y = task.evaluate(np.array(new_x))
    X = np.append(X, new_x, axis=0)
    Y = np.append(Y, new_y, axis=0)

    # Visualize the objective function, model and the acquisition function
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1 = plot_objective_function(task, ax1, X, Y)
    ax1 = plot_model(model, task.X_lower, task.X_upper, ax1)
    ax2 = plot_acquisition_function(acquisition_func, task.X_lower,
                                    task.X_upper, ax2)
    plt.show(block=True)
