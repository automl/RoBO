
import numpy as np


class BaseMaximizer(object):

    def __init__(self, objective_function, X_lower, X_upper, rng=None):
        """
        Interface for optimizers that maximizing the
        acquisition function.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        X_lower: np.ndarray (D)
            Lower bounds of the input space
        X_upper: np.ndarray (D)
            Upper bounds of the input space
        rng: numpy.random.RandomState
            Random number generator
        """
        self.X_lower = X_lower
        self.X_upper = X_upper
        self.objective_func = objective_function
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(10000))
        else:
            self.rng = rng

    def maximize(self):
        pass
