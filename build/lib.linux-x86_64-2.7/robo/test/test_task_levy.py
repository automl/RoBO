'''
Created on 14.07.2015

@author: aaron
'''
import setup_logger
import unittest
import numpy as np
from robo.task.synthetic_functions.levy import Levy


class TestTaskBranin(unittest.TestCase):

    def test_branin(self):
        levy = Levy()

        # Check batch computation
        n_points = 10
        X = np.random.rand(n_points, levy.n_dims)
        X[:, 0] = X[:, 0].dot(levy.X_upper[0] - levy.X_lower[0]) + levy.X_lower[0]
        y = levy.evaluate(X)

        assert len(y.shape) == 2
        assert y.shape[0] == n_points
        assert y.shape[1] == 1

        # Check single computation
        X = np.array([np.random.rand(levy.n_dims)])

        X[:, 0] = X[:, 0].dot(levy.X_upper[0] - levy.X_lower[0]) + levy.X_lower[0]

        y = levy.evaluate(X)
        assert y.shape[0] == 1

        # Check optimas
        X = levy.opt
        y = levy.evaluate(X)

        assert np.all(np.round(y, 6) == np.array([levy.fopt]))

if __name__ == "__main__":
    unittest.main()
