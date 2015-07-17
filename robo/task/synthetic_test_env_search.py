'''
Created on 08.07.2015

@author: Aaron Klein
'''

import time
import numpy as np

from robo.task.base_task import BaseTask


class SyntheticFktEnvSearch(BaseTask):
    '''
    classdocs
    '''

    def __init__(self):
        self.X_lower = np.array([0.0, 0.01])
        self.X_upper = np.array([1, 1])
        self.n_dims = 2
        self.is_env = np.array([0, 1])

    def objective_function(self, x):

        y = 5 - x[:, 0] * np.log(x[:, 1]) + x[:, 0] / 2.
        time.sleep(x[0, 1])
        return np.array([y])
