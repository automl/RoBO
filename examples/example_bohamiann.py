import logging
import numpy as np

from robo.fmin.bayesian_optimization import bayesian_optimization

logging.basicConfig(level=logging.INFO)


def objective_function(x):
    y = np.sin(3 * x[0]) * 4 * (x[0] - 1) * (x[0] + 2)
    return y


# defining the bounds and dimensions of the input space
lower = np.array([0])
upper = np.array([6])

# start Bayesian optimization to minimize the objective function
results = bayesian_optimization(objective_function, lower, upper, model_type="bohamiann", num_iterations=20)
print(results["x_opt"])
print(results["f_opt"])
