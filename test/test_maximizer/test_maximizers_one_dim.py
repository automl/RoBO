
import unittest
import numpy as np

from robo.maximizers.cmaes import CMAES
from robo.maximizers.direct import Direct
from robo.maximizers.grid_search import GridSearch
from robo.maximizers.stochastic_local_search import StochasticLocalSearch


def objective_function(x):
    y = (0.5 - x) ** 2
    return y[0]


class TestMaximizers1D(unittest.TestCase):

    def setUp(self):

        self.lower = np.array([0])
        self.upper = np.array([1])

    def test_direct(self):
        maximizer = Direct(objective_function, self.lower, self.upper)
        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert np.all(x >= self.lower)
        assert np.all(x <= self.upper)

    def test_stochastic_local_search(self):
        maximizer = StochasticLocalSearch(objective_function, self.lower, self.upper)
        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert np.all(x >= self.lower)
        assert np.all(x <= self.upper)

    def test_grid_search(self):
        maximizer = GridSearch(objective_function, self.lower, self.upper)
        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert np.all(x >= self.lower)
        assert np.all(x <= self.upper)


def objective_function(x):
    y = (0.5 - x) ** 2
    return y[0]

class TestMaximizers2D(unittest.TestCase):

    def setUp(self):

        self.branin = Branin()

        n_points = 5
        rng = np.random.RandomState(42)
        self.X = init_random_uniform(self.branin.X_lower,
                                     self.branin.X_upper,
                                     n_points,
                                     rng=rng)

        self.Y = self.branin.evaluate(self.X)

        kernel = GPy.kern.Matern52(input_dim=self.branin.n_dims)
        self.model = GPyModel(kernel, optimize=True,
                              noise_variance=1e-4,
                              num_restarts=10)

        self.model.train(self.X, self.Y)
        self.acquisition_func = EI(self.model,
                                   X_upper=self.branin.X_upper,
                                   X_lower=self.branin.X_lower,
                                   par=0.1)

    def test_direct(self):
        maximizer = Direct(self.acquisition_func,
                           self.branin.X_lower,
                           self.branin.X_upper)
        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert x.shape[1] == self.branin.n_dims
        assert np.all(x[:, 0] >= self.branin.X_lower[0])
        assert np.all(x[:, 1] >= self.branin.X_lower[1])
        assert np.all(x[:, 0] <= self.branin.X_upper[0])
        assert np.all(x[:, 1] <= self.branin.X_upper[1])

    def test_stochastic_local_search(self):
        maximizer = StochasticLocalSearch(self.acquisition_func,
                                          self.branin.X_lower,
                                          self.branin.X_upper)
        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert x.shape[1] == self.branin.n_dims
        assert np.all(x[:, 0] >= self.branin.X_lower[0])
        assert np.all(x[:, 1] >= self.branin.X_lower[1])
        assert np.all(x[:, 0] <= self.branin.X_upper[0])
        assert np.all(x[:, 1] <= self.branin.X_upper[1])

    def test_cmaes(self):
        maximizer = CMAES(self.acquisition_func,
                          self.branin.X_lower,
                          self.branin.X_upper)

        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert x.shape[1] == self.branin.n_dims
        assert np.all(x[:, 0] >= self.branin.X_lower[0])
        assert np.all(x[:, 1] >= self.branin.X_lower[1])
        assert np.all(x[:, 0] <= self.branin.X_upper[0])
        assert np.all(x[:, 1] <= self.branin.X_upper[1])


if __name__ == "__main__":
    unittest.main()
