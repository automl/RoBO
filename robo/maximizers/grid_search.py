'''
Created on 13.07.2015

@author: Aaron Klein
'''
import numpy as np

from robo.maximizers.base_maximizer import BaseMaximizer


class GridSearch(BaseMaximizer):

    def __init__(self, objective_function, X_lower, X_upper, resolution=1000):
        """
        Evaluates a equally spaced grid to maximize the acquisition function
        in a one dimensional input space.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        X_lower: np.ndarray (D)
            Lower bounds of the input space
        X_upper: np.ndarray (D)
            Upper bounds of the input space
        resolution: int
            Defines of how many data points the grid consists.
        """
        self.resolution = resolution
        if X_lower.shape[0] > 1:
            raise RuntimeError("Grid search works just for \
                one dimensional functions")
        super(GridSearch, self).__init__(objective_function, X_lower, X_upper)

    def maximize(self):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """

        x = np.linspace(self.X_lower[0], self.X_upper[0],
                    self.resolution).reshape((self.resolution, 1, 1))
        # y = array(map(acquisition_fkt, x))
        ys = np.zeros([self.resolution])
        for i in range(self.resolution):
            ys[i] = self.objective_func(x[i])
        y = np.array(ys)
        x_star = x[y.argmax()]
        return x_star
