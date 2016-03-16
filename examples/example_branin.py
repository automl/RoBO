'''
Created on Jun 23, 2015

@author: Aaron Klein
'''

import setup_logger

import GPy
from robo.models.gpy_model import GPyModel
from robo.acquisition.ei import EI
from robo.maximizers.cmaes import CMAES
from robo.task.synthetic_functions.branin import Branin
from robo.solver.bayesian_optimization import BayesianOptimization


branin = Branin()

kernel = GPy.kern.Matern52(input_dim=branin.n_dims)
model = GPyModel(kernel)

acquisition_func = EI(model,
                     X_upper=branin.X_upper,
                     X_lower=branin.X_lower,
                     par=0.1)

maximizer = CMAES(acquisition_func, branin.X_lower, branin.X_upper)

bo = BayesianOptimization(acquisition_func=acquisition_func,
                          model=model,
                          maximize_func=maximizer,
                          task=branin)

bo.run(10)
