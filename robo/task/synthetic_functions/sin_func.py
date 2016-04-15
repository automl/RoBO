'''
Created on Dec 15, 2015

@author: Aaron Klein
'''

import numpy as np

from robo.task.base_task import BaseTask


class SinFunction(BaseTask):

    def __init__(self):
        X_lower = np.array([0])
        X_upper = np.array([7])

        # Estimated via grid search
        #opt = np.array([[0.82808280828082814]])
        #fopt = np.array([[-148.66741422]])
        super(SinFunction, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        y = 150 + np.sin(3 * x) * 4 * (x - 1) * (x + 2)
        return y
