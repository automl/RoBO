'''
Created on 12.07.2015

@author: Aaron Klein
'''
import numpy as np

from robo.task.base_task import BaseTask


class Branin(BaseTask):

    def __init__(self):
        X_lower = np.array([-5, 0])
        X_upper = np.array([10, 15])
        opt = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        fopt = 0.397887
        super(Branin, self).__init__(X_lower, X_upper, opt, fopt)

    def objective_function(self, x):
        y = (x[:, 1] - (5.1 / (4 * np.pi ** 2)) * x[:, 0] ** 2 + 5 * x[:, 0] / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[:, 0]) + 10

        return y[:, np.newaxis]
