import unittest
import numpy as np

from robo.task.branin import Branin
from robo.task.hartmann6 import Hartmann6
from robo.task.goldstein_price import GoldsteinPrice


class TestBenchmarks(unittest.TestCase):

    def test_branin(self):
        task = Branin()

        # Check batch computation
        n_points = 10
        X = np.random.rand(n_points, task.n_dims)
        X[:, 0] = X[:, 0].dot(task.X_upper[0] - task.X_lower[0]) + task.X_lower[0]
        X[:, 1] = X[:, 1].dot(task.X_upper[1] - task.X_lower[1]) + task.X_lower[1]
        y = task.evaluate(X)
        assert y.shape[0] == n_points
        assert y.shape[1] == 1

        # Check single computation
        X = np.array([np.random.rand(task.n_dims)])

        X[:, 0] = X[:, 0].dot(task.X_upper[0] - task.X_lower[0]) + task.X_lower[0]
        X[:, 1] = X[:, 1].dot(task.X_upper[1] - task.X_lower[1]) + task.X_lower[1]

        y = task.evaluate(X)
        assert y.shape[0] == 1

        # Check optimas
        X = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        y = task.evaluate(X)

        assert np.all(np.round(y, 6) == 0.397887) == True

    def test_hartmann6(self):
        task = Hartmann6()

        # Check batch computation
        n_points = 10
        X = np.random.rand(n_points, task.n_dims)
        y = task.evaluate(X)

        assert y.shape[1] == 1
        assert y.shape[0] == n_points

        # Check single computation
        X = np.array([np.random.rand(task.n_dims)])

        y = task.evaluate(X)
        assert y.shape[0] == 1

        # Check optimas
        X = np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])
        y = task.evaluate(X)

        assert np.all(np.round(y, 5) == -3.32237) == True

    def test_goldstein_price(self):
        task = GoldsteinPrice()

        # Check batch computation
        n_points = 10
        X = np.random.rand(n_points, task.n_dims)

        X[:, 0] = X[:, 0].dot(task.X_upper[0] - task.X_lower[0]) + task.X_lower[0]
        X[:, 1] = X[:, 1].dot(task.X_upper[1] - task.X_lower[1]) + task.X_lower[1]
        y = task.evaluate(X)
        assert y.shape[0] == n_points
        assert y.shape[1] == 1

        # Check single computation
        X = np.array([np.random.rand(task.n_dims)])

        X[:, 0] = X[:, 0].dot(task.X_upper[0] - task.X_lower[0]) + task.X_lower[0]
        X[:, 1] = X[:, 1].dot(task.X_upper[1] - task.X_lower[1]) + task.X_lower[1]

        y = task.evaluate(X)
        assert y.shape[0] == 1

        # Check optimas
        X = np.array([[0, -1]])
        y = task.evaluate(X)

        assert np.all(np.round(y) == 3) == True


if __name__ == "__main__":
    unittest.main()
