'''
Created on 12.07.2015

@author: Aaron Klein
'''

import numpy as np


class BaseTask(object):
    '''
    classdocs
    '''

    def __init__(self, X_lower, X_upper, opt=None, fopt=None, do_scaling=True):
        '''
        Constructor
        '''
        self.X_lower = X_lower
        self.X_upper = X_upper
        self.n_dims = self.X_lower.shape[0]
        assert self.n_dims == self.X_upper.shape[0]
        self.opt = opt
        self.fopt = fopt

        if do_scaling:
            self.original_X_lower = self.X_lower
            self.original_X_upper = self.X_upper
            self.X_lower = -1 * np.ones(self.original_X_lower.shape)
            self.X_upper = 1 * np.ones(self.original_X_upper.shape)
            self.do_scaling = True

    def objective_function(self, x):
        pass

    def evaluate(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_dims
        assert np.all(x >= self.X_lower)
        assert np.all(x <= self.X_upper)

        if self.do_scaling:
            x = (self.original_X_upper - self.original_X_lower) * (x - self.X_lower) / (self.X_upper - self.X_lower) + self.original_X_lower
        return self.objective_function(x)
