'''
Created on 14.07.2015

@author: aaron
'''
import setup_logger
import unittest
import numpy as np

from robo.task.synthetic_functions.hartmann6 import Hartmann6


class TestTaskHartmann6(unittest.TestCase):

    def test_hartmann6(self):
        hartmann6 = Hartmann6()

        # Check batch computation
        n_points = 10
        X = np.random.rand(n_points, hartmann6.n_dims)
        y = hartmann6.evaluate(X)

        assert y.shape[1] == 1
        assert y.shape[0] == n_points
        assert len(y.shape) == 2

        # Check single computation
        X = np.array([np.random.rand(hartmann6.n_dims)])

        y = hartmann6.evaluate(X)
        assert y.shape[0] == 1

        # Check optimas
        X = hartmann6.opt
        y = hartmann6.evaluate(X)

        assert np.all(np.round(y, 5) == hartmann6.fopt) == True


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()