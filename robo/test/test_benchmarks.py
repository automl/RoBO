import unittest
import numpy as np

from robo.benchmarks.branin import branin, get_branin_bounds
from robo.benchmarks.hartmann6 import hartmann6, get_hartmann6_bounds
from robo.benchmarks.goldstein_price import goldstein_price, get_goldstein_price_bounds


class TestBenchmarks(unittest.TestCase):

    def test_branin(self):
        X_lower, X_upper, n_dims = get_branin_bounds()

        # Check batch computation
        n_points = 10
        X = np.random.rand(n_points, n_dims)
        X[:, 0] = X[:, 0].dot(X_upper[0] - X_lower[0]) + X_lower[0]
        X[:, 1] = X[:, 1].dot(X_upper[1] - X_lower[1]) + X_lower[1]
        y = branin(X)
        assert y.shape[0] == n_points

        # Check single computation
        X = np.array([np.random.rand(n_dims)])

        X[:, 0] = X[:, 0].dot(X_upper[0] - X_lower[0]) + X_lower[0]
        X[:, 1] = X[:, 1].dot(X_upper[1] - X_lower[1]) + X_lower[1]

        y = branin(X)
        assert y.shape[0] == 1

        # Check optimas
        X = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        y = branin(X)

        assert np.all(np.round(y, 6) == 0.397887) == True

    def test_hartmann6(self):
        _, _, n_dims = get_hartmann6_bounds()

        # Check batch computation
        n_points = 10
        X = np.random.rand(n_points, n_dims)
        y = hartmann6(X)

        assert y.shape[0] == n_points

        # Check single computation
        X = np.array([np.random.rand(n_dims)])

        y = hartmann6(X)
        assert y.shape[0] == 1

        # Check optimas
        X = np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])
        y = hartmann6(X)

        assert np.all(np.round(y, 5) == -3.32237) == True

    def test_goldstein_price(self):
        X_lower, X_upper, n_dims = get_goldstein_price_bounds()

        # Check batch computation
        n_points = 10
        X = np.random.rand(n_points, n_dims)

        X[:, 0] = X[:, 0].dot(X_upper[0] - X_lower[0]) + X_lower[0]
        X[:, 1] = X[:, 1].dot(X_upper[1] - X_lower[1]) + X_lower[1]
        y = goldstein_price(X)
        assert y.shape[0] == n_points

        # Check single computation
        X = np.array([np.random.rand(n_dims)])

        X[:, 0] = X[:, 0].dot(X_upper[0] - X_lower[0]) + X_lower[0]
        X[:, 1] = X[:, 1].dot(X_upper[1] - X_lower[1]) + X_lower[1]

        y = goldstein_price(X)
        assert y.shape[0] == 1

        # Check optimas
        X = np.array([[0, -1]])
        y = goldstein_price(X)

        assert np.all(np.round(y) == 3) == True


if __name__ == "__main__":
    unittest.main()
