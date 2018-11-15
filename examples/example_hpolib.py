"""
This example shows how RoBO can be combined with HPOlib.
Before you run it, make sure that you installed it.
For further information have a look here https://github.com/automl/HPOlib2.git
"""
import numpy as np
from hpolib.benchmarks.synthetic_functions import Branin

from robo.fmin import bayesian_optimization

f = Branin()
info = f.get_meta_information()
bounds = np.array(info['bounds'])

# Start Bayesian optimization to optimize the objective function
results = bayesian_optimization(f, bounds[:, 0], bounds[:, 1], num_iterations=50)
print(results["x_opt"])
print(results["f_opt"])
