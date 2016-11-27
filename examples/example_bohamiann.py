import logging
import numpy as np

from hpolib.benchmarks.synthetic_functions import Branin

from robo.fmin import bohamiann

logging.basicConfig(level=logging.INFO)

f = Branin()
info = f.get_meta_information()
bounds = np.array(info['bounds'])

# Start Bayesian optimization to optimize the objective function
results = bohamiann(f, bounds[:, 0], bounds[:, 1], num_iterations=20)
print(results["x_opt"])
print(results["f_opt"])
