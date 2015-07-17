'''
Created on 12.07.2015

@author: Aaron Klein
'''

import numpy as np


class BaseTask(object):
    '''
    classdocs
    '''

    def __init__(self, X_lower, X_upper, opt=None, fopt=None):
        '''
        Constructor
        '''
        self.X_lower = X_lower
        self.X_upper = X_upper
        self.n_dims = self.X_lower.shape[0]
        assert self.n_dims == self.X_upper.shape[0]
        self.opt = opt
        self.fopt = fopt

    def objective_function(self, x):
        pass

    def evaluate(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_dims
        assert np.all(x[:, 0] >= self.X_lower[0])
        assert np.all(x[:, 1] >= self.X_lower[1])
        assert np.all(x[:, 0] <= self.X_upper[0])
        assert np.all(x[:, 1] <= self.X_upper[1])
        assert np.all(x < self.X_upper)

        return self.objective_function(x)
