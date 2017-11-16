import sys
import numpy as np

from scipy import optimize
from functools import partial

from robo.maximizers.base_maximizer import BaseMaximizer
from robo.initial_design import init_random_uniform


class SciPyOptimizer(BaseMaximizer):

    def __init__(self, objective_function, lower,
                 upper, n_restarts=10, verbosity=False, rng=None):
        """
        Interface for scipy's L-BFGS-B implementation.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        lower: np.ndarray (D)
            Lower bounds of the input space
        upper: np.ndarray (D)
            Upper bounds of the input space
        n_restarts: int
            Determines how often the local search is repeated.
        verbosity: bool
            Show scipy output.

        """
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        self.n_restarts = n_restarts
        self.verbosity = verbosity
        super(SciPyOptimizer, self).__init__(objective_function,
                                             lower, upper)

    def _acquisition_fkt_wrapper(self, x, acq_f):

        if np.any(np.isnan(x)):
            return sys.float_info.max

        a = -acq_f(np.array([np.clip(x, self.lower, self.upper)]))[0]

        if np.any(np.isinf(a)):
            return sys.float_info.max
        else:
            return a

    def maximize(self):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(D,)
            Point with highest acquisition value.
        """
        cand = []
        cand_vals = []

        f = partial(self._acquisition_fkt_wrapper, acq_f=self.objective_func)

        starts = init_random_uniform(self.lower, self.upper, int(self.n_restarts * 0.5))
        rand_incs = np.array([np.random.normal(loc=self.objective_func.model.get_incumbent()[0],
                                               scale=np.ones([self.lower.shape[0]]) * 0.5)
                              for _ in range(int(self.n_restarts * 0.5))])
        starts = np.append(starts, rand_incs, axis=0)

        for start in starts:

           res = optimize.minimize(f, start, method='L-BFGS-B',
                                   bounds=list(zip(self.lower, self.upper)),
                                   options={"disp": self.verbosity})

           cand.append(res["x"])
           cand_vals.append(res["fun"])

        best = np.argmin(cand_vals)

        return np.clip(cand[best], self.lower, self.upper)
