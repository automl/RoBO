
import numpy as np

from robo.fmin import fmin


# The optimization function that we want to optimize.
# It gets a numpy array with shape (N,D) where N >= 1 are the number of
# datapoints and D are the number of features
def objective_function(x):
    y = np.sin(3 * x) * 4 * (x - 1) * (x + 2)
    return y

# Defining the bounds and dimensions of the input space
X_lower = np.array([0])
X_upper = np.array([6])

# Start Bayesian optimization to optimize the objective function
results = fmin(objective_function, X_lower, X_upper)
print(results["x_opt"])
print(results["f_opt"])

