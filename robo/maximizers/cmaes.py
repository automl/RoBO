'''
Created on 13.07.2015

@author: Aaron Klein
'''
import sys
import cma
import StringIO
import numpy as np

from base_maximizer import BaseMaximizer


class CMAES(BaseMaximizer):
    '''
     Wrapper around the python implementation of the Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES)
    '''

    def __init__(self, objective_function, X_lower, X_upper):
        if X_lower.shape[0] == 1:
            raise RuntimeError("CMAES does not works in a one dimensional function space")
        super(CMAES, self).__init__(objective_function, X_lower, X_upper)

    def _cma_fkt_wrapper(self, acq_f, derivative=False):
        def _l(x, *args, **kwargs):
            x = np.array([x])
            return -acq_f(x, derivative=derivative, *args, **kwargs)[0]
        return _l

    def maximize(self, verbose=False):
        if not verbose:
            # Turn off stdout, a bit hacky but that's the only way how we get cma to be quiet
            stdout = sys.stdout
            sys.stdout = StringIO.StringIO()

            # Start from incumbent if the objective function has one otherwise start from random
            if (hasattr(self.objective_func, "compute_incumbent")):
                start_point, _ = self.objective_func.compute_incumbent(self.objective_func.model, self.X_lower, self.X_upper)
            else:
                start_point = np.random.uniform(self.X_lower, self.X_upper, self.X_lower.shape[0])

            res = cma.fmin(self._cma_fkt_wrapper(self.objective_func),
                           start_point, 0.6,
                           options={"bounds": [self.X_lower, self.X_upper], "verbose": 0, "verb_log": sys.maxint})

            # Turn on stdout
            sys.stdout = stdout
        else:
            res = cma.fmin(self._cma_fkt_wrapper(self.objective_func), (self.X_upper + self.X_lower) * 0.5,
                           0.6, options={"bounds": [self.X_lower, self.X_upper], "verbose": 0, "verb_log": sys.maxint})
        return np.array([res[0]])
