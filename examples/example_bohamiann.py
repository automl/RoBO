import logging
import numpy as np

from hpolib.benchmarks.synthetic_functions import Branin

from robo.fmin import bohamiann

logging.basicConfig(level=logging.INFO)


def objective_function(x):
    y = np.sin(3 * x[0]) * 4 * (x[0] - 1) * (x[0] + 2)
    return y

# Defining the bounds and dimensions of the input space
lower = np.array([0])
upper = np.array([6])

# Start Bayesian optimization to optimize the objective function
results = bohamiann(objective_function, lower, upper, num_iterations=20)
print(results["x_opt"])
print(results["f_opt"])
