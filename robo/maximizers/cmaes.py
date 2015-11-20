'''
Created on 13.07.2015

@author: Aaron Klein
'''
import sys
import logging
import cma

import numpy as np

from robo.maximizers.base_maximizer import BaseMaximizer


class CMAES(BaseMaximizer):

    def __init__(self, objective_function, X_lower, X_upper):
        """
        Interface for the  Covariance Matrix Adaptation Evolution Strategy
        python package

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
        """
        if X_lower.shape[0] == 1:
            raise RuntimeError("CMAES does not works in a one \
                dimensional function space")

        super(CMAES, self).__init__(objective_function, X_lower, X_upper)

    def _cma_fkt_wrapper(self, acq_f, derivative=False):
        def _l(x, *args, **kwargs):
            x = np.array([x])
            return -acq_f(x, derivative=derivative, *args, **kwargs)[0]
        return _l

    def maximize(self):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """
        res = cma.fmin(
            self._cma_fkt_wrapper(
                self.objective_func),
            (self.X_upper + self.X_lower) * 0.5,
            0.6,
            options={
                "bounds": [
                    self.X_lower,
                    self.X_upper],
                "verbose": 0,
                "verb_log": sys.maxsize})
        if res[0] is None:
            logging.error("CMA-ES did not find anything. \
                Return random configuration instead.")
            return np.array([np.random.uniform(self.X_lower,
                                               self.X_upper,
                                               self.X_lower.shape[0])])
        return np.array([res[0]])
