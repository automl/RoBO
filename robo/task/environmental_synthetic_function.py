'''
Created on Oct 28, 2015

@author: Aaron Klein
'''

import numpy as np

from robo.task.base_task import BaseTask


class SyntheticFunctionWrapper(BaseTask):

    def __init__(self, original_task):
        self.original_task = original_task

        # Add an additional dimension for the system size
        X_lower = np.concatenate((self.original_task.original_X_lower,
                                  np.array([0])))

        X_upper = np.concatenate((self.original_task.original_X_upper,
                                  np.array([1])))

        self.is_env = np.zeros([self.original_task.n_dims])
        self.is_env = np.concatenate((self.is_env, np.ones([1])))

        super(SyntheticFunctionWrapper, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        y = self.original_task.objective_function(x[:, self.is_env == 0]) \
            * np.exp(-(x[0, -1] - 1))

        cost = np.exp(x[:, self.is_env == 1])

        return y, cost

    def objective_function_test(self, x):
        return self.original_task.objective_function(x[:, self.is_env == 0])


class NoisySyntheticFunction(BaseTask):

    def __init__(self, original_task, sigma_min, sigma_max,
                 c_min, c_max,
                 k_cost, k_noise):

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.c_min = c_min
        self.c_max = c_max
        self.k_cost = k_cost
        self.k_noise = k_noise

        self.original_task = original_task

        X_lower = np.concatenate((self.original_task.original_X_lower,
                                  np.array([0])))

        X_upper = np.concatenate((self.original_task.original_X_upper,
                                  np.array([1])))

        self.is_env = np.zeros([self.original_task.n_dims])
        self.is_env = np.concatenate((self.is_env, np.ones([1])))

        super(NoisySyntheticFunction, self).__init__(X_lower, X_upper)

    def sigma_function(self, s):
        sigma = self.sigma_min
        sigma += (self.sigma_max - self.sigma_min) * ((1 - s) ** self.k_noise)
        return sigma

    def cost_function(self, s):
        cost = self.c_min + (self.c_max - self.c_min) * (s ** self.k_cost)
        return cost

    def objective_function(self, x):
        s = x[:, self.is_env == 1]

        y = self.original_task.objective_function(x[:, self.is_env == 0])
        y += self.sigma_function(s) * np.random.randn()

        cost = self.cost_function(s)

        return y, cost

    def objective_function_test(self, x):
        return self.original_task.objective_function(x[:, self.is_env == 0])
