'''
Created on 12.07.2015

@author: Aaron Klein
'''
import numpy as np

from robo.task.base_task import BaseTask


class Bohachevsky(BaseTask):

    def __init__(self):
        X_lower = np.array([-100, -100])
        X_upper = np.array([100, 100])
        opt = np.array([[0, 0]])
        fopt = 0.0
        super(Bohachevsky, self).__init__(X_lower, X_upper, opt, fopt)

    def objective_function(self, x):
        y = 0.7 + x[:, 0] ** 2 + 2.0 * x[:, 1] ** 2
        y -= 0.3 * np.cos(3.0 * np.pi * x[:, 0])
        y -= 0.4 * np.cos(4.0 * np.pi * x[:, 1])
        return y[:, np.newaxis]

    def objective_function_test(self, x):
        return self.objective_function(x)
