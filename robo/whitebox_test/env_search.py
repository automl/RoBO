'''
Created on Jun 29, 2015

@author: Aaron Klein
'''

import GPy

from robo.models.GPyModel import GPyModel
from robo.solver.env_bayesian_optimization import EnvBayesianOptimization
from robo.acquisition.EnvEntropySearch import EnvEntropySearch
from robo.maximizers.maximize import direct
from robo.recommendation.incumbent import compute_incumbent
from robo.benchmarks.branin_with_costs import branin_with_costs, get_branin_with_costs_bounds


X_lower, X_upper, n_dims, is_env_variable = get_branin_with_costs_bounds()


kernel = GPy.kern.Matern52(input_dim=n_dims)
model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

cost_kernel = GPy.kern.Matern52(input_dim=n_dims)
cost_model = GPyModel(cost_kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

n_representer = 10
n_hals_vals = 100
n_func_samples = 200


acquisition_func = EnvEntropySearch(model, cost_model, X_lower=X_lower, X_upper=X_upper,
                                    is_env_variable=is_env_variable, n_representer=n_representer,
                                    n_hals_vals=n_hals_vals, n_func_samples=n_func_samples, compute_incumbent=compute_incumbent)
maximizer = direct

bo = EnvBayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          cost_model=cost_model,
                          maximize_fkt=maximizer,
                          X_lower=X_lower,
                          X_upper=X_upper,
                          dims=n_dims,
                          objective_fkt=branin_with_costs,
                          save_dir="./out_env_search",
                          num_save=1)

bo.run(num_iterations=10)
