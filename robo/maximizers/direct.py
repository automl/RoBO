'''
Created on 13.07.2015

@author: Aaron Klein
'''

import DIRECT

import numpy as np

from robo.maximizers.base_maximizer import BaseMaximizer


class Direct(BaseMaximizer):
    '''
    classdocs
    '''

    def __init__(self, objective_function, X_lower, X_upper,
                 n_func_evals=1000, n_iters=2000):

        self.n_func_evals = n_func_evals
        self.n_iters = n_iters
        super(Direct, self).__init__(objective_function, X_lower, X_upper)

    def _direct_acquisition_fkt_wrapper(self, acq_f):
        def _l(x, user_data):
            return -acq_f(np.array([x])), 0
        return _l

    def maximize(self):
        x, fmin, ierror = DIRECT.solve(
            self._direct_acquisition_fkt_wrapper(
                self.objective_func), l=[
                self.X_lower], u=[
                self.X_upper], maxT=self.n_iters, maxf=self.n_func_evals)
        return np.array([x])
