'''
Created on 12.07.2015

@author: Aaron Klein
'''
import numpy as np

from robo.task.goldstein_price import GoldsteinPrice
from robo.task.base_task import BaseTask


class EnvGoldsteinPrice(BaseTask):

    def __init__(self):
        self.goldstein_price = GoldsteinPrice()
        X_lower = np.concatenate((self.goldstein_price.original_X_lower, np.array([0])))
        X_upper = np.concatenate((self.goldstein_price.original_X_upper, np.array([1])))
        self.is_env = np.array([0, 0, 1])
        super(EnvGoldsteinPrice, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        y = self.goldstein_price.objective_function(x[:, :-1]) * np.exp(-(x[0, -1] - 1))

        return y

    def objective_function_test(self, x):
        return self.goldstein_price.objective_function(x[:, :-1])
