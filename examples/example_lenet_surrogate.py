'''
Created on Jul 1, 2015

@author: Aaron Klein
'''


import GPy
from robo.models.GPyModel import GPyModel
from robo.acquisition.EntropyMC import EntropyMC
from robo.maximizers.maximize import direct
from robo.recommendation.incumbent import compute_incumbent
from robo.benchmarks.lenet_mnist_surrogate import lenet_mnist_surrogate, get_lenet_mnist_surrogate_bounds
from robo.bayesian_optimization import BayesianOptimization


X_lower, X_upper, dims = get_lenet_mnist_surrogate_bounds()

maximizer = direct

kernel = GPy.kern.Matern52(input_dim=dims)
model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

acquisition_func = EntropyMC(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1)

bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          maximize_fkt=maximizer,
                          X_lower=X_lower,
                          X_upper=X_upper,
                          dims=dims,
                          objective_fkt=lenet_mnist_surrogate,
                          save_dir="./test_output",
                          num_save=1)

bo.run(10)
