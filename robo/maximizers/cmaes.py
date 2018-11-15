import sys
import logging
try:
    import cma
except ImportError:
    raise ImportError("""
        In order to use this module, CMA need to be installed. Try running
        pip install cma
    """)

from robo.maximizers.base_maximizer import BaseMaximizer
from robo.initial_design.init_random_uniform import init_random_uniform


class CMAES(BaseMaximizer):

    def __init__(self, objective_function, lower, upper,
                 verbose=True, restarts=0, n_func_evals=1000, rng=None):
        """
        Interface for the  Covariance Matrix Adaptation Evolution Strategy
        python package

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
        verbose: bool
            If set to False the CMAES output is disabled
        restarts: int
            Number of restarts for CMAES
        rng: numpy.random.RandomState
            Random number generator
        """
        if lower.shape[0] == 1:
            raise RuntimeError("CMAES does not works in a one \
                dimensional function space")

        super(CMAES, self).__init__(objective_function, lower, upper, rng)

        self.restarts = restarts
        self.verbose = verbose
        self.n_func_evals = n_func_evals

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

        start_point = init_random_uniform(self.lower, self.upper, 1, self.rng)

        def obj_func(x):
            a = self.objective_func(x[None, :])
            return -a[0]

        res = cma.fmin(obj_func, x0=start_point[0], sigma0=0.6,
                       restarts=self.restarts,
                       options={"bounds": [self.lower, self.upper],
                                "verbose": verbose_level,
                                "verb_log": sys.maxsize,
                                "maxfevals": self.n_func_evals})
        if res[0] is None:
            logging.error("CMA-ES did not find anything. \
                Return random configuration instead.")
            return start_point

        return res[0]
