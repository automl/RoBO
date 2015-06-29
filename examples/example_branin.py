'''
Created on Jun 23, 2015

@author: Aaron Klein
'''

import GPy
from robo.models.GPyModel import GPyModel
from robo.acquisition.EntropyMC import EntropyMC
from robo.maximizers.maximize import cmaes
from robo.recommendation.incumbent import compute_incumbent
from robo.benchmarks.branin import branin, get_branin_bounds
from robo.bayesian_optimization import BayesianOptimization


X_lower, X_upper, dims = get_branin_bounds()

maximizer = cmaes

kernel = GPy.kern.Matern52(input_dim=dims)
model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

acquisition_func = EntropyMC(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1)

bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          maximize_fkt=maximizer,
                          X_lower=X_lower,
                          X_upper=X_upper,
                          dims=dims,
                          objective_fkt=branin,
                          save_dir="./test_output",
                          num_save=1)

bo.run(10)
