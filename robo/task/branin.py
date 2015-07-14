'''
Created on 12.07.2015

@author: Aaron Klein
'''
import numpy as np

from robo.task.base_task import BaseTask


class Branin(BaseTask):
    '''
    classdocs
    '''

    def __init__(self):
        self.X_lower = np.array([-5, 0])
        self.X_upper = np.array([10, 15])
        self.n_dims = 2
        self.opt = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        self.fopt = 0.397887

    def objective_function(self, x):
        y = (x[:, 1] - (5.1 / (4 * np.pi ** 2)) * x[:, 0] ** 2 + 5 * x[:, 0] / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[:, 0]) + 10

        return y[:, np.newaxis]
