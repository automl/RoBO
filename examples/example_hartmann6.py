'''
Created on Jun 23, 2015

@author: Aaron Klein
'''

import GPy
from robo.models.GPyModel import GPyModel
from robo.acquisition.EntropyMC import EntropyMC
from robo.maximizers.maximize import stochastic_local_search
from robo.benchmarks.hartmann6 import hartmann6, get_hartmann6_bounds
from robo.bayesian_optimization import BayesianOptimization
from robo.recommendation.incumbent import compute_incumbent

X_lower, X_upper, dims = get_hartmann6_bounds()

maximizer = stochastic_local_search

kernel = GPy.kern.Matern52(input_dim=dims)
model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

acquisition_func = EntropyMC(model, X_lower, X_upper, compute_incumbent)

bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          maximize_fkt=maximizer,
                          X_lower=X_lower,
                          X_upper=X_upper,
                          dims=dims,
                          objective_fkt=hartmann6,
                          save_dir="./test_output",
                          num_save=1)

bo.run(10)
