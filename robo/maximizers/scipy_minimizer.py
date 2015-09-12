'''
Created on Jul 30, 2015

@author: Aaron Klein
'''

import numpy as np

from scipy import optimize

from robo.maximizers.base_maximizer import BaseMaximizer
from _functools import partial


class SciPyMinimizer(BaseMaximizer):
    def __init__(self, objective_function, X_lower, X_upper, n_local_searches=10):
        self.n_local_searches = n_local_searches
        super(SciPyMinimizer, self).__init__(objective_function, X_lower, X_upper)

    def _direct_acquisition_fkt_wrapper(self, x, acq_f):
        print x

        return -acq_f(np.array([x]))

    def maximize(self, verbosity=True):
        cand = np.zeros([self.n_local_searches, self.X_lower.shape[0]])
        cand_vals = np.zeros([self.n_local_searches])

        f = partial(self._direct_acquisition_fkt_wrapper, acq_f=self.objective_func)

        for i in range(self.n_local_searches):
            start = np.array([np.random.uniform(self.X_lower, self.X_upper, self.X_lower.shape[0])])
            print "start %s" % (start)
            res = optimize.minimize(f, start, method="L-BFGS-B", bounds=zip(self.X_lower, self.X_upper), options={"disp": verbosity})
            cand[i] = res["x"]
            cand_vals[i] = res["fun"]
            print "cand %s, cand_val %f" % (cand[i], cand_vals[i])
        best = np.argmax(cand_vals)
        return np.array([cand[best]])
