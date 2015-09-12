'''
Created on 12.07.2015

@author: Aaron Klein
'''
import numpy as np

from robo.task.hartmann6 import Hartmann6


class EnvHartmann6(Hartmann6):

    def __init__(self):
        super(EnvHartmann6, self).__init__()
        self.X_lower = np.concatenate((self.X_lower, np.array([0])))
        self.X_upper = np.concatenate((self.X_upper, np.array([1])))
        self.is_env = np.array([0, 0, 0, 0, 0, 0, 1])
        self.n_dims = 7

    def objective_function(self, x):
        #eps = np.random.exponential(x[0, -1])
        y = super(EnvHartmann6, self).objective_function(x[:, :-1]) * np.exp(-(x[0, -1] - 1))

        return y

    def evaluate_test(self, x):
        return self.objective_function(x[:, :-1])
