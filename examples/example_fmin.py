'''
Created on Jul 3, 2015

@author: Aaron Klein
'''
import numpy as np

from robo.fmin import fmin


# The optimization function that we want to optimize.
# It gets a numpy array with shape (N,D) where N >= 1 are the number of
# datapoints and D are the number of features
def objective_function(x):
    return np.sin(3 * x) * 4 * (x - 1) * (x + 2)

# Defining the bounds and dimensions of the input space
X_lower = np.array([0])
X_upper = np.array([6])

# Start Bayesian optimization to optimize the objective function
x_best, fval = fmin(objective_function, X_lower, X_upper)
