'''
Created on 14.07.2015

@author: aaron
'''
import unittest
import numpy as np

from robo.task.synthetic_functions.hartmann3 import Hartmann3
from robo.task.synthetic_functions.sin import SinOne
from robo.task.synthetic_functions.sin import SinTwo

from robo.initial_design.init_random_uniform import init_random_uniform


class TestTaskHartmann3(unittest.TestCase):

    def test(self):
        task = Hartmann3()

        # Check batch computation
        n_points = 10
        X = init_random_uniform(task.X_lower, task.X_upper, n_points)
        y = task.evaluate(X)

        assert len(y.shape) == 2
        assert y.shape[0] == n_points
        assert y.shape[1] == 1

        # Check single computation
        X = init_random_uniform(task.X_lower, task.X_upper, 1)

        y = task.evaluate(X)
        assert y.shape[0] == 1

        # Check optimas
        X = task.opt
        y = task.evaluate(X)

        assert np.all(np.round(y, 6) == np.array([task.fopt]))


class TestTaskSinOne(unittest.TestCase):

    def test(self):
        task = SinOne()

        # Check batch computation
        n_points = 10
        X = init_random_uniform(task.X_lower, task.X_upper, n_points)
        y = task.evaluate(X)

        assert len(y.shape) == 2
        assert y.shape[0] == n_points
        assert y.shape[1] == 1

        # Check single computation
        X = init_random_uniform(task.X_lower, task.X_upper, 1)

        y = task.evaluate(X)
        assert y.shape[0] == 1

        # Check optimas
        #X = task.opt
        #y = task.evaluate(X)

        #assert np.all(np.round(y, 6) == np.array([task.fopt]))


class TestTaskSinTwo(unittest.TestCase):

    def test(self):
        task = SinTwo()

        # Check batch computation
        n_points = 10
        X = init_random_uniform(task.X_lower, task.X_upper, n_points)
        y = task.evaluate(X)

        assert len(y.shape) == 2
        assert y.shape[0] == n_points
        assert y.shape[1] == 1

        # Check single computation
        X = init_random_uniform(task.X_lower, task.X_upper, 1)

        y = task.evaluate(X)
        assert y.shape[0] == 1

        # Check optimas
        #X = task.opt
        #y = task.evaluate(X)

        #assert np.all(np.round(y, 6) == np.array([task.fopt]))


if __name__ == "__main__":
    unittest.main()
