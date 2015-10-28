'''
Created on Oct 28, 2015

@author: Aaron Klein
'''

import numpy as np

from robo.task.base_task import BaseTask


class EnvironmentalSyntheticFunction(BaseTask):

    def __init__(self, original_task):
        self.original_task = original_task

        X_lower = np.concatenate((self.original_task.original_X_lower,
                                  np.array([0])))

        X_upper = np.concatenate((self.original_task.original_X_upper,
                                  np.array([1])))

        self.is_env = np.zeros([self.original_task.n_dims])
        self.is_env = np.concatenate((self.is_env, np.ones([1])))

        super(EnvironmentalSyntheticFunction, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        y = self.original_task.objective_function(x[:, self.is_env == 0]) \
                    * np.exp(-(x[0, -1] - 1))

        cost = np.exp(x[:, self.is_env == 1])

        return y, cost

    def objective_function_test(self, x):
        return self.original_task.objective_function(x[:, self.is_env == 0])
