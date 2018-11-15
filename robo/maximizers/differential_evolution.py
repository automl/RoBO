import sys
import numpy as np
import scipy as sp

from robo.maximizers.base_maximizer import BaseMaximizer


class DifferentialEvolution(BaseMaximizer):

    def __init__(self, objective_function, lower, upper, n_iters=20, rng=None):
        """

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        lower: np.ndarray (D)
            Lower bounds of the input space
        upper: np.ndarray (D)
            Upper bounds of the input space
        n_iters: int
            Number of iterations
        """
        self.n_iters = n_iters
        super(DifferentialEvolution, self).__init__(objective_function, lower, upper, rng)

    def _acquisition_fkt_wrapper(self, acq_f):
        def _l(x):
            a = -acq_f(np.array([np.clip(x, self.lower, self.upper)]))
            if np.any(np.isinf(a)):
                return sys.float_info.max
            return a

        return _l

    def maximize(self):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """

        bounds = list(zip(self.lower, self.upper))

        res = sp.optimize.differential_evolution(self._acquisition_fkt_wrapper(self.objective_func),
                                                 bounds, maxiter=self.n_iters)

        return np.clip(res["x"], self.lower, self.upper)
