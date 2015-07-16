'''
Created on 13.07.2015

@author: Aaron Klein
'''
import sys
import StringIO
import numpy as np

from base_maximizer import BaseMaximizer


class GridSearch(BaseMaximizer):
    '''
    classdocs
    '''

    def __init__(self, objective_function, X_lower, X_upper, resolution=1000):
        self.resolution = resolution
        if X_lower.shape[0] > 1:
            raise RuntimeError("Grid search works just for one dimensional functions")
        super(GridSearch, self).__init__(objective_function, X_lower, X_upper)

    def maximize(self, verbose=False):
        x = np.linspace(self.X_lower[0], self.X_upper[0], self.resolution).reshape((self.resolution, 1, 1))
        # y = array(map(acquisition_fkt, x))
        ys = np.zeros([self.resolution])
        for i in range(self.resolution):
            ys[i] = self.objective_func(x[i])
        y = np.array(ys)
        x_star = x[y.argmax()]
        return x_star
