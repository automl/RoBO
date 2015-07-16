'''
Created on 16.07.2015

@author: Aaron Klein
'''

import GPy
import unittest
import numpy as np

from robo.models.GPyModel import GPyModel
from robo.recommendation.incumbent import compute_incumbent
from robo.recommendation.optimize_posterior import optimize_posterior_mean,\
    optimize_posterior_mean_and_std


class Test(unittest.TestCase):

    def func(self, x):
        return x**x + 0.5

    def setUp(self):
        X = np.array([[0.0, 0.25, 0.75, 1.0]])
        X = X.T
        y = self.func(X)
        k = GPy.kern.RBF(input_dim=1)
        self.m = GPyModel(k, noise_variance=1e-3)
        self.m.train(X, y)
        self.X_lower = np.array([0])
        self.X_upper = np.array([1.0])

    def tearDown(self):
        pass

    def test_optimize_posterior_mean(self):
        inc, inc_val = optimize_posterior_mean(self.m, self.X_lower, self.X_upper, with_gradients=True)

        assert len(inc.shape) == 1
        assert np.all(inc >= self.X_lower)
        assert np.all(inc <= self.X_upper)
        assert np.all(inc_val <= compute_incumbent(self.m)[1])

    def test_optimize_posterior_mean_and_std(self):
        inc, inc_val = optimize_posterior_mean_and_std(self.m, self.X_lower, self.X_upper, with_gradients=True)

        assert len(inc.shape) == 1
        assert np.all(inc >= self.X_lower)
        assert np.all(inc <= self.X_upper)
        assert np.all(inc_val <= compute_incumbent(self.m)[1])


if __name__ == "__main__":
    unittest.main()
