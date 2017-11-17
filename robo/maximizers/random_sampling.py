import numpy as np

from robo.maximizers.base_maximizer import BaseMaximizer
from robo.initial_design import init_random_uniform


class RandomSampling(BaseMaximizer):

    def __init__(self, objective_function, lower, upper, n_samples=500, rng=None):
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

        # Sample random points uniformly over the whole space
        rand = init_random_uniform(self.lower, self.upper,
                                   int(self.n_samples * .7))

        # Put a Gaussian on the incumbent and sample from that
        loc = self.objective_func.model.get_incumbent()[0],
        scale = np.ones([self.lower.shape[0]]) * 0.1
        rand_incs = np.array([np.clip(np.random.normal(loc, scale), self.lower, self.upper)[0]
                              for _ in range(int(self.n_samples * 0.3))])

        X = np.concatenate((rand, rand_incs), axis=0)
        y = self.objective_func(X)

        x_star = X[y.argmax()]

        return x_star
