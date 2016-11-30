import numpy as np

from robo.maximizers.base_maximizer import BaseMaximizer


class GridSearch(BaseMaximizer):

    def __init__(self, objective_function, lower, upper, resolution=1000):
        """
        Evaluates a equally spaced grid to maximize the acquisition function
        in a one dimensional input space.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        lower: np.ndarray (D)
            Lower bounds of the input space
        upper: np.ndarray (D)
            Upper bounds of the input space
        resolution: int
            Defines of how many data points the grid consists.
        """
        self.resolution = resolution
        if lower.shape[0] > 1:
            raise RuntimeError("Grid search works just for \
                one dimensional functions")
        super(GridSearch, self).__init__(objective_function, lower, upper)

    def maximize(self):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """

        x = np.linspace(self.lower[0], self.upper[0], self.resolution).reshape((self.resolution, 1, 1))
        # y = array(map(acquisition_fkt, x))
        ys = np.zeros([self.resolution])
        for i in range(self.resolution):
            ys[i] = self.objective_func(x[i])
        y = np.array(ys)
        x_star = x[y.argmax()]

        return x_star[0]
