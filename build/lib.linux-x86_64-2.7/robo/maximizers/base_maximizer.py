'''
Created on 13.07.2015

@author: Aaron Klein
'''


class BaseMaximizer(object):

    def __init__(self, objective_function, X_lower, X_upper):
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
        """
        self.X_lower = X_lower
        self.X_upper = X_upper
        self.objective_func = objective_function

    def maximize(self):
        pass
