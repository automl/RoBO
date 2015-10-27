'''
Created on 12.07.2015

@author: Aaron Klein
'''
import numpy as np

from robo.task.base_task import BaseTask
from robo.task.bohachevsky import Bohachevsky


class EnvBohachevsky(BaseTask):
    def __init__(self):
        self.bohachevsky = Bohachevsky()
        X_lower = np.concatenate((self.bohachevsky.original_X_lower, np.array([0])))
        X_upper = np.concatenate((self.bohachevsky.original_X_upper, np.array([1])))
        self.is_env = np.array([0, 0, 1])
        super(EnvBohachevsky, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        y = self.bohachevsky.objective_function(x[:, :-1]) * np.exp(-(x[0, -1] - 1))

        return y

    def objective_function_test(self, x):
        return self.bohachevsky.objective_function(x[:, :-1])
