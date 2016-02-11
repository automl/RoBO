'''
Created on 13.07.2015

@author: Aaron Klein
'''

import DIRECT

import numpy as np

from robo.maximizers.base_maximizer import BaseMaximizer


class Direct(BaseMaximizer):

    def __init__(self, objective_function, X_lower, X_upper,
                 n_func_evals=400, n_iters=200):
        """
        Interface for the DIRECT algorithm by D. R. Jones, C. D. Perttunen
        and B. E. Stuckmann

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        X_lower: np.ndarray (D)
            Lower bounds of the input space
        X_upper: np.ndarray (D)
            Upper bounds of the input space
        n_func_evals: int
            The maximum number of function evaluations
        n_iters: int
            The maximum number of iterations
        """
        self.n_func_evals = n_func_evals
        self.n_iters = n_iters
        super(Direct, self).__init__(objective_function, X_lower, X_upper)

    def _direct_acquisition_fkt_wrapper(self, acq_f):
        def _l(x, user_data):
            return -acq_f(np.array([x])), 0
        return _l

    def maximize(self):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """
        x, _, _ = DIRECT.solve(
            self._direct_acquisition_fkt_wrapper(self.objective_func),
                               l=[self.X_lower],
                               u=[self.X_upper],
                               maxT=self.n_iters,
                               maxf=self.n_func_evals)
        return np.array([x])
