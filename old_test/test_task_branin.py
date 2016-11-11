'''
Created on 14.07.2015

@author: aaron
'''
import setup_logger
import unittest
import numpy as np
from robo.task.synthetic_functions.branin import Branin


class TestTaskBranin(unittest.TestCase):

    def test_branin(self):
        branin = Branin()

        # Check batch computation
        n_points = 10
        X = np.random.rand(n_points, branin.n_dims)
        X[:, 0] = X[:, 0].dot(branin.X_upper[0] - branin.X_lower[0]) + branin.X_lower[0]
        X[:, 1] = X[:, 1].dot(branin.X_upper[1] - branin.X_lower[1]) + branin.X_lower[1]

        y = branin.evaluate(X)

        assert len(y.shape) == 2
        assert y.shape[0] == n_points
        assert y.shape[1] == 1

        # Check single computation
        X = np.array([np.random.rand(branin.n_dims)])

        X[:, 0] = X[:, 0].dot(branin.X_upper[0] - branin.X_lower[0]) + branin.X_lower[0]
        X[:, 1] = X[:, 1].dot(branin.X_upper[1] - branin.X_lower[1]) + branin.X_lower[1]

        y = branin.evaluate(X)
        assert y.shape[0] == 1

        # Check optimas
        X = branin.opt
        y = branin.evaluate(X)

        assert np.all(np.round(y, 6) == np.array([branin.fopt]))

if __name__ == "__main__":
    unittest.main()
