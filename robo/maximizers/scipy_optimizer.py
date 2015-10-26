'''
Created on Jul 30, 2015

@author: Aaron Klein
'''

import numpy as np

from scipy import optimize

from robo.maximizers.base_maximizer import BaseMaximizer
from _functools import partial


class SciPyOptimizer(BaseMaximizer):
    
    def __init__(self, objective_function, X_lower, X_upper, n_restarts=10, verbosity=False):
        self.n_restarts = n_restarts
        self.verbosity = verbosity
        super(SciPyOptimizer, self).__init__(objective_function, X_lower, X_upper)

    def _direct_acquisition_fkt_wrapper(self, x, acq_f):
        return -acq_f(np.array([x]))

    def maximize(self):
        cand = np.zeros([self.n_restarts, self.X_lower.shape[0]])
        cand_vals = np.zeros([self.n_restarts])

        f = partial(self._direct_acquisition_fkt_wrapper, acq_f=self.objective_func)

        for i in range(self.n_restarts):
            start = np.array([np.random.uniform(self.X_lower, self.X_upper, self.X_lower.shape[0])])
            
            res = optimize.minimize(f, start, method="L-BFGS-B", bounds=zip(self.X_lower, self.X_upper), options={"disp": self.verbosity})
            cand[i] = res["x"]
            cand_vals[i] = res["fun"]
            
        best = np.argmax(cand_vals)
        return np.array([cand[best]])


class SciPyGlobalOptimizer(BaseMaximizer):
    
    def __init__(self, objective_function, X_lower, X_upper, n_restarts=10, verbosity=False):
        self.n_restarts = n_restarts
        self.verbosity = verbosity

        super(SciPyGlobalOptimizer, self).__init__(objective_function, X_lower, X_upper)

    def _direct_acquisition_fkt_wrapper(self, x, acq_f):
        return -acq_f(np.array([x]))

    def maximize(self):
        cand = np.zeros([self.n_restarts, self.X_lower.shape[0]])
        cand_vals = np.zeros([self.n_restarts])

        f = partial(self._direct_acquisition_fkt_wrapper, acq_f=self.objective_func)

        for i in range(self.n_restarts):
            start = np.array([np.random.uniform(self.X_lower, self.X_upper, self.X_lower.shape[0])])
            res = optimize.basinhopping(f, start, minimizer_kwargs={"bounds" : zip(self.X_lower, self.X_upper), "method": "L-BFGS-B"}, disp=self.verbosity)
     
            cand[i] = res.x
            cand_vals[i] = res.fun
        best = np.argmax(cand_vals)
        return np.array([cand[best]])
