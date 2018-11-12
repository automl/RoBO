"""
This example shows how RoBO can be combined with the fANOVA package.
Before you run it, make sure that you installed the fANOVA package:
https://github.com/automl/fanova.git
"""
import os

import fanova.visualizer
import numpy as np
from fanova import fANOVA
from hpolib.benchmarks.synthetic_functions import Branin

from robo.fmin import random_search

objective_function = Branin()
info = objective_function.get_meta_information()
bounds = np.array(info['bounds'])
config_space = objective_function.get_configuration_space()

# Start Bayesian optimization to optimize the objective function
results = random_search(objective_function, bounds[:, 0], bounds[:, 1], num_iterations=50)

# Creating a fANOVA object
X = np.array([i for i in results['X']])
Y = np.array([i for i in results['y']])
f = fANOVA(X, Y)

print(f.quantify_importance((0,)))

# Visualization
os.makedirs("./plots", exist_ok=True)
vis = fanova.visualizer.Visualizer(f, config_space, "./plots/")
vis.plot_marginal(1)
