'''
Created on Jun 26, 2015

@author: Aaron Klein
'''

import GPy
import unittest
import numpy as np

from robo.models.GPyModel import GPyModel
from robo.maximizers.cmaes import CMAES
from robo.maximizers.direct import Direct
from robo.maximizers.grid_search import GridSearch
from robo.maximizers.stochastic_local_search import StochasticLocalSearch
from robo.acquisition.EI import EI
from robo.recommendation.incumbent import compute_incumbent
from robo.task.branin import Branin


def objective_function(x):
    return np.sin(3 * x) * 4 * (x - 1) * (x + 2)


class TestMaximizers1D(unittest.TestCase):

    def setUp(self):

        self.X_lower = np.array([0])
        self.X_upper = np.array([6])
        self.dims = 1

        n_points = 5
        self.X = np.random.rand(n_points, self.dims)

        self.X[:, 0] = self.X[:, 0].dot(self.X_upper[0] - self.X_lower[0]) + self.X_lower[0]

        self.Y = objective_function(self.X)

        kernel = GPy.kern.Matern52(input_dim=self.dims)
        self.model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

        self.model.train(self.X, self.Y)
        self.acquisition_func = EI(self.model, X_upper=self.X_upper, X_lower=self.X_lower, compute_incumbent=compute_incumbent, par=0.1)

    def test_direct(self):
        maximizer = Direct(self.acquisition_func, self.X_lower, self.X_upper)
        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert x.shape[1] == self.dims
        assert np.all(x[:, 0] >= self.X_lower[0])
        assert np.all(x[:, 0] <= self.X_upper[0])
        assert np.all(x < self.X_upper)

    def test_stochastic_local_search(self):
        maximizer = StochasticLocalSearch(self.acquisition_func, self.X_lower, self.X_upper)
        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert x.shape[1] == self.dims
        assert np.all(x[:, 0] >= self.X_lower[0])
        assert np.all(x[:, 0] <= self.X_upper[0])
        assert np.all(x < self.X_upper)

    def test_grid_search(self):
        maximizer = GridSearch(self.acquisition_func, self.X_lower, self.X_upper)
        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert x.shape[1] == self.dims
        assert np.all(x[:, 0] >= self.X_lower[0])
        assert np.all(x[:, 0] <= self.X_upper[0])
        assert np.all(x < self.X_upper)


class TestMaximizers2D(unittest.TestCase):

    def setUp(self):

        self.branin = Branin()

        n_points = 5
        self.X = np.random.rand(n_points, self.branin.n_dims)

        self.X[:, 0] = self.X[:, 0].dot(self.branin.X_upper[0] - self.branin.X_lower[0]) + self.branin.X_lower[0]
        self.X[:, 1] = self.X[:, 1].dot(self.branin.X_upper[1] - self.branin.X_lower[1]) + self.branin.X_lower[1]

        self.Y = self.branin.evaluate(self.X)

        kernel = GPy.kern.Matern52(input_dim=self.branin.n_dims)
        self.model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

        self.model.train(self.X, self.Y)
        self.acquisition_func = EI(self.model,
                                   X_upper=self.branin.X_upper, X_lower=self.branin.X_lower,
                                   compute_incumbent=compute_incumbent, par=0.1)

    def test_direct(self):
        maximizer = Direct(self.acquisition_func, self.branin.X_lower, self.branin.X_upper)
        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert x.shape[1] == self.branin.n_dims
        assert np.all(x[:, 0] >= self.branin.X_lower[0])
        assert np.all(x[:, 1] >= self.branin.X_lower[1])
        assert np.all(x[:, 0] <= self.branin.X_upper[0])
        assert np.all(x[:, 1] <= self.branin.X_upper[1])
        assert np.all(x < self.branin.X_upper)

    def test_stochastic_local_search(self):
        maximizer = StochasticLocalSearch(self.acquisition_func, self.branin.X_lower, self.branin.X_upper)
        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert x.shape[1] == self.branin.n_dims
        assert np.all(x[:, 0] >= self.branin.X_lower[0])
        assert np.all(x[:, 1] >= self.branin.X_lower[1])
        assert np.all(x[:, 0] <= self.branin.X_upper[0])
        assert np.all(x[:, 1] <= self.branin.X_upper[1])
        assert np.all(x < self.branin.X_upper)

    def test_cmaes(self):
        maximizer = CMAES(self.acquisition_func, self.branin.X_lower, self.branin.X_upper)
        x = maximizer.maximize(verbose=False)

        assert x.shape[0] == 1
        assert x.shape[1] == self.branin.n_dims
        assert np.all(x[:, 0] >= self.branin.X_lower[0])
        assert np.all(x[:, 1] >= self.branin.X_lower[1])
        assert np.all(x[:, 0] <= self.branin.X_upper[0])
        assert np.all(x[:, 1] <= self.branin.X_upper[1])
        assert np.all(x < self.branin.X_upper)


if __name__ == "__main__":
    unittest.main()
