import os
import sys

try:
    import DIRECT
except ImportError:
    raise ImportError("""
        In order to use this module, DIRECT need to be installed. Try running
        pip install direct
    """)

import numpy as np

from robo.maximizers.base_maximizer import BaseMaximizer


class Direct(BaseMaximizer):

    def __init__(self, objective_function, lower, upper,
                 n_func_evals=400, n_iters=200, verbose=True):
        """
        Interface for the DIRECT algorithm by D. R. Jones, C. D. Perttunen
        and B. E. Stuckmann

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        lower: np.ndarray (D)
            Lower bounds of the input space
        upper: np.ndarray (D)
            Upper bounds of the input space
        n_func_evals: int
            The maximum number of function evaluations
        n_iters: int
            The maximum number of iterations
        verbose: bool
            Suppress Direct's output.
        """
        self.n_func_evals = n_func_evals
        self.n_iters = n_iters
        self.verbose = verbose

        super(Direct, self).__init__(objective_function, lower, upper)

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
        if self.verbose:
            x, _, _ = DIRECT.solve(self._direct_acquisition_fkt_wrapper(self.objective_func),
                                   l=[self.lower],
                                   u=[self.upper],
                                   maxT=self.n_iters,
                                   maxf=self.n_func_evals)
        else:
            fileno = sys.stdout.fileno()
            with os.fdopen(os.dup(fileno), 'wb') as stdout:
                with os.fdopen(os.open(os.devnull, os.O_WRONLY), 'wb') as devnull:
                    sys.stdout.flush();
                    os.dup2(devnull.fileno(), fileno)  # redirect
                    x, _, _ = DIRECT.solve(self._direct_acquisition_fkt_wrapper(self.objective_func),
                                           l=[self.lower],
                                           u=[self.upper],
                                           maxT=self.n_iters,
                                           maxf=self.n_func_evals)
                sys.stdout.flush();
                os.dup2(stdout.fileno(), fileno)  # restore
        return x
