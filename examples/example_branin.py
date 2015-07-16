'''
Created on Jun 23, 2015

@author: Aaron Klein
'''

import GPy
from robo.models.GPyModel import GPyModel
from robo.acquisition.EI import EI
from robo.maximizers.cmaes import CMAES
from robo.recommendation.incumbent import compute_incumbent
from robo.task.branin import Branin
from robo.solver.bayesian_optimization import BayesianOptimization


branin = Branin()

kernel = GPy.kern.Matern52(input_dim=branin.n_dims)
model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

acquisition_func = EI(model,
                     X_upper=branin.X_upper,
                     X_lower=branin.X_lower,
                     compute_incumbent=compute_incumbent,
                     par=0.1)

maximizer = CMAES(acquisition_func, branin.X_lower, branin.X_upper)

bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          maximize_fkt=maximizer,
                          task=branin,
                          save_dir="./test_output",
                          num_save=1)

bo.run(10)
