'''
Created on 12.07.2015

@author: Aaron Klein
'''
import numpy as np

from robo.task.base_task import BaseTask


class Levy(BaseTask):

    def __init__(self):
        X_lower = np.array([-15])
        X_upper = np.array([10])
        opt = np.array([[1.0]])
        fopt = 0.0
        super(Levy, self).__init__(X_lower, X_upper, opt=opt, fopt=fopt)

    def objective_function(self, x):
        z = 1 + ((x - 1.) / 4.)
        s = np.power((np.sin(np.pi * z)), 2)
        y = (s + ((z - 1) ** 2) * (1 + np.power((np.sin(2 * np.pi * z)), 2)))

        return y

    def objective_function_test(self, x):
        return self.objective_function(x)
