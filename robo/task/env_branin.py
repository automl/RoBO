'''
Created on 12.07.2015

@author: Aaron Klein
'''
import numpy as np

from robo.task.base_task import BaseTask
from robo.task.branin import Branin


class EnvBranin(BaseTask):

    def __init__(self):
        self.branin = Branin()
        X_lower = np.concatenate((self.branin.original_X_lower, np.array([0])))
        X_upper = np.concatenate((self.branin.original_X_upper, np.array([1])))
        self.is_env = np.array([0, 0, 1])
        super(EnvBranin, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        y = self.branin.objective_function(x[:, :-1]) * np.exp(-(x[0, -1] - 1))
        
        return y

    def objective_function_test(self, x):
        return self.branin.objective_function(x[:, :-1])
