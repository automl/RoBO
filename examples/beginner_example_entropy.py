'''
Created on Apr 17, 2015

@author: Aaron Klein
'''
import GPy
import numpy as np

from robo.models.GPyModel import GPyModel
from robo.acquisition.EI import EI
from robo.acquisition.Entropy import Entropy
from robo.maximizers.maximize import grid_search
from robo.recommendation.incumbent import compute_incumbent

from robo import BayesianOptimization


def objective_function(x):
        return  np.sin(3 * x) * 4 * (x - 1) * (x + 2)

X_lower = np.array([0])
X_upper = np.array([6])

dims = 1

kernel = GPy.kern.Matern52(input_dim=dims)
model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)
proposal_measurement = EI(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1)
acquisition_func = Entropy(model, X_lower, X_upper, sampling_acquisition=proposal_measurement)
maximizer = grid_search

bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          maximize_fkt=maximizer,
                          X_lower=X_lower,
                          X_upper=X_upper,
                          dims=dims,
                          objective_fkt=objective_function)

bo.run(num_iterations=10)
