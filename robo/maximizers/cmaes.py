'''
Created on 13.07.2015

@author: Aaron Klein
'''
import sys
import os
import logging
import cma

import numpy as np

from robo.maximizers.base_maximizer import BaseMaximizer


class CMAES(BaseMaximizer):

    def __init__(self, objective_function, X_lower, X_upper,
                 verbose=True, restarts=0, n_func_evals=1000):
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
        verbose: bool
            If set to False the CMAES output is disabled
        restarts: int
            Number of restarts for CMAES
        """
        if X_lower.shape[0] == 1:
            raise RuntimeError("CMAES does not works in a one \
                dimensional function space")

        super(CMAES, self).__init__(objective_function, X_lower, X_upper)

        self.restarts = restarts
        self.verbose = verbose
        self.n_func_evals = n_func_evals

    def _cma_fkt_wrapper(self, acq_f, derivative=False):
        def _l(x, *args, **kwargs):
            x = np.array([x])
            return -acq_f(x, derivative=derivative, *args, **kwargs)[0]
        return _l

    def maximize(self, rng = None):
        """
        Maximizes the given acquisition function.

        Parameters
        ----------


        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """

        # All stdout and stderr is pointed to devnull during
        # the optimization. (That is the only way to keep cmaes quiet)
        if rng is None:
            rng = np.random.RandomState(42)
        if not self.verbose:
            sys.stdout = os.devnull
            sys.stderr = os.devnull
        res = cma.fmin(self._cma_fkt_wrapper(self.objective_func),
                x0=(self.X_upper + self.X_lower) * 0.5, sigma0=0.6,
                restarts=self.restarts,
                options={"bounds": [self.X_lower, self.X_upper],
                         "verbose": 0,
                         "verb_log": sys.maxsize,
                         "maxfevals": self.n_func_evals})
        if res[0] is None:
            logging.error("CMA-ES did not find anything. \
                Return random configuration instead.")
            return np.array([rng.uniform(self.X_lower,
                                               self.X_upper,
                                               self.X_lower.shape[0])])
        if not self.verbose:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        return np.array([res[0]])
