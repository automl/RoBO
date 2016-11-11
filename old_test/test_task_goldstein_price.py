'''
Created on 14.07.2015

@author: Aaron Klein
'''
import setup_logger
import unittest
import numpy as np

from robo.task.synthetic_functions.goldstein_price import GoldsteinPrice


class TestTaskGoldsteinPrice(unittest.TestCase):

        goldstein_price = GoldsteinPrice()

        # Check batch computation
        n_points = 10
        X = np.random.rand(n_points, goldstein_price.n_dims)

        X[:, 0] = X[:, 0].dot(goldstein_price.X_upper[0] - goldstein_price.X_lower[0]) + goldstein_price.X_lower[0]
        X[:, 1] = X[:, 1].dot(goldstein_price.X_upper[1] - goldstein_price.X_lower[1]) + goldstein_price.X_lower[1]
        y = goldstein_price.evaluate(X)
        assert y.shape[0] == n_points
        assert y.shape[1] == 1
        assert len(y.shape) == 2

        # Check single computation
        X = np.array([np.random.rand(goldstein_price.n_dims)])

        X[:, 0] = X[:, 0].dot(goldstein_price.X_upper[0] - goldstein_price.X_lower[0]) + goldstein_price.X_lower[0]
        X[:, 1] = X[:, 1].dot(goldstein_price.X_upper[1] - goldstein_price.X_lower[1]) + goldstein_price.X_lower[1]

        y = goldstein_price.evaluate(X)
        assert y.shape[0] == 1

        # Check optimas
        X = goldstein_price.opt
        y = goldstein_price.evaluate(X)

        assert np.all(np.round(y) == goldstein_price.fopt) == True


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()