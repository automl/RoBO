
import numpy as np

from scipy import optimize

from robo.maximizers.base_maximizer import BaseMaximizer
from functools import partial


class SciPyOptimizer(BaseMaximizer):

    def __init__(self, objective_function, X_lower,
                 X_upper, n_restarts=10, verbosity=False, rng=None):
        """
        Interface for scipy's LBFGS implementation.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        X_lower: np.ndarray (D)
            Lower bounds of the input space
        X_upper: np.ndarray (D)
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
                                             X_lower, X_upper)

    def _acquisition_fkt_wrapper(self, x, acq_f):
        return -acq_f(np.array([x]))

    def maximize(self):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """
        cand = np.zeros([self.n_restarts, self.X_lower.shape[0]])
        cand_vals = np.zeros([self.n_restarts])

        f = partial(self._acquisition_fkt_wrapper, acq_f=self.objective_func)

        for i in range(self.n_restarts):
            start = np.array([self.rng.uniform(self.X_lower,
                                                self.X_upper,
                                                self.X_lower.shape[0])])

            res = optimize.minimize(
                f,
                start,
                method="L-BFGS-B",
                bounds=zip(
                    self.X_lower,
                    self.X_upper),
                options={
                    "disp": self.verbosity})
            cand[i] = res["x"]
            cand_vals[i] = res["fun"]

        best = np.argmax(cand_vals)
        return np.array([cand[best]])


class SciPyGlobalOptimizer(BaseMaximizer):

    def __init__(self, objective_function, X_lower, X_upper,
                 n_restarts=10, verbosity=False, n_func_evals=200, rng=None):
        """
        Interface for scipy's global optimization method.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        X_lower: np.ndarray (D)
            Lower bounds of the input space
        X_upper: np.ndarray (D)
            Upper bounds of the input space
        n_restarts: int
            Determines how often the local search is repeated.
        verbosity: bool
            Show scipy output.
        """
        self.n_restarts = n_restarts
        self.verbosity = verbosity
        self.n_func_evals = n_func_evals
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        super(SciPyGlobalOptimizer, self).__init__(objective_function,
                                                   X_lower, X_upper)

    def _acquisition_fkt_wrapper(self, x, acq_f):
        print x
        return -acq_f(np.array([x]))

    def maximize(self):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """
        cand = np.zeros([self.n_restarts, self.X_lower.shape[0]])
        cand_vals = np.zeros([self.n_restarts])

        f = partial(self._acquisition_fkt_wrapper, acq_f=self.objective_func)

        for i in range(self.n_restarts):
            start = np.array([self.rng.uniform(self.X_lower,
                                                self.X_upper,
                                                self.X_lower.shape[0])])
            res = optimize.basinhopping(
                f,
                start,
                niter=self.n_func_evals,
                minimizer_kwargs={
                    "bounds": zip(
                        self.X_lower,
                        self.X_upper),
                    "method": "L-BFGS-B"},
                disp=self.verbosity)

            cand[i] = res.x
            cand_vals[i] = res.fun
        best = np.argmax(cand_vals)
        print np.array([cand[best]])
        return np.array([cand[best]])
