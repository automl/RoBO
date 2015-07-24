'''
Created on Jul 23, 2015

@author: Aaron Klein
'''

import GPy
import numpy as np

from robo.task.rembo import REMBO
from robo.task.branin import Branin
from robo.models.GPyModel import GPyModel
from robo.maximizers.cmaes import CMAES
from robo.recommendation.incumbent import compute_incumbent
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.acquisition.EI import EI


class BraninInBillionDims(REMBO):
    def __init__(self):
        self.b = Branin()
        X_lower = np.concatenate((self.b.X_lower, np.zeros([999998])))
        X_upper = np.concatenate((self.b.X_upper, np.ones([999998])))
        super(BraninInBillionDims, self).__init__(X_lower, X_upper, d=2)

    def objective_function(self, x):
        return self.b.objective_function(x[:, :2])

task = BraninInBillionDims()
kernel = GPy.kern.Matern52(input_dim=task.n_dims)
model = GPyModel(kernel, optimize=True, noise_variance=1e-3, num_restarts=10)
acquisition_func = EI(model, task.X_lower, task.X_upper, compute_incumbent)
maximizer = CMAES(acquisition_func, task.X_lower, task.X_upper)
bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                      model=model,
                      maximize_fkt=maximizer,
                      task=task)

bo.run(500)
