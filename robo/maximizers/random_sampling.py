from robo.maximizers.base_maximizer import BaseMaximizer
from robo.initial_design import init_random_uniform


class RandomSampling(BaseMaximizer):

    def __init__(self, objective_function, lower, upper, n_samples=1000, rng=None):
        """
        Samples candidates uniformly at random and returns the point with the highest objective value.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        lower: np.ndarray (D)
            Lower bounds of the input space
        upper: np.ndarray (D)
            Upper bounds of the input space
        n_samples: int
            Number of candidates that are samples
        """
        self.n_samples = n_samples
        super(RandomSampling, self).__init__(objective_function, lower, upper, rng)

    def maximize(self):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """

        X = init_random_uniform(self.lower,
                                self.upper,
                                self.n_samples)

        y = self.objective_func(X)

        x_star = X[y.argmax()]

        return x_star
