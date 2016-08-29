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
from robo.initial_design.init_random_uniform import init_random_uniform


class CMAES(BaseMaximizer):

    def __init__(self, objective_function, X_lower, X_upper,
                 verbose=True, restarts=0, n_func_evals=1000, rng=None):
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
        rng: numpy.random.RandomState
            Random number generator
        """
        if X_lower.shape[0] == 1:
            raise RuntimeError("CMAES does not works in a one \
                dimensional function space")

        super(CMAES, self).__init__(objective_function, X_lower, X_upper, rng)

        self.restarts = restarts
        self.verbose = verbose
        self.n_func_evals = n_func_evals

    # def _cma_fkt_wrapper(self, acq_f):
    #     def _l(x, *args, **kwargs):
    #         x = np.array([x])
    #         return -acq_f(x, *args, **kwargs)[0]
    #     res = _l
    #     print res
    #     return res

    def maximize(self):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with the highest acquisition value.
        """

        verbose_level = -9
        if self.verbose:
            verbose_level = 0

        start_point = init_random_uniform(self.X_lower, self.X_upper, 1, self.rng)

        def obj_func(x):
            a = self.objective_func(x)[0, 0]
            return a

        res = cma.fmin(obj_func,
                x0=start_point[0],
                sigma0=0.6,
                restarts=self.restarts,
                options={"bounds": [self.X_lower, self.X_upper],
                         "verbose": verbose_level,
                         "verb_log": sys.maxsize,
                         "maxfevals": self.n_func_evals})
        if res[0] is None:
            logging.error("CMA-ES did not find anything. \
                Return random configuration instead.")
            return start_point

        return np.array([res[0]])
