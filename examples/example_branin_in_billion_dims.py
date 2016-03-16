'''
Created on Jul 23, 2015

@author: Aaron Klein
'''

import GPy
import numpy as np

from robo.task.rembo import REMBO
from robo.task.synthetic_functions.branin import Branin
from robo.models.gpy_model import GPyModel
from robo.maximizers.cmaes import CMAES
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.acquisition.ei import EI


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
model = GPyModel(kernel, optimize=True, num_restarts=10)
acquisition_func = EI(model, task.X_lower, task.X_upper)
maximizer = CMAES(acquisition_func, task.X_lower, task.X_upper)
bo = BayesianOptimization(acquisition_func=acquisition_func,
                      model=model,
                      maximize_func=maximizer,
                      task=task)

bo.run(500)
