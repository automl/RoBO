'''
Created on 12.07.2015

@author: Aaron Klein
'''
import numpy as np

from robo.task.base_task import BaseTask
from robo.task.hartmann6 import Hartmann6


class EnvHartmann6(BaseTask):

    def __init__(self):
        self.hartmann6 = Hartmann6()
        X_lower = np.concatenate((self.hartmann6.original_X_lower, np.array([0])))
        X_upper = np.concatenate((self.hartmann6.original_X_upper, np.array([1])))
        self.is_env = np.array([0, 0, 0, 0, 0, 0, 1])
        super(EnvHartmann6, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        y = self.hartmann6.objective_function(x[:, :-1]) * np.exp(-(x[0, -1] - 1))

        return y

    def objective_function_test(self, x):
        return self.hartmann6.objective_function(x[:, :-1])
