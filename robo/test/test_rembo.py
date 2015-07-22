'''
Created on Jul 21, 2015

@author: Aaron Klein
'''
import unittest
import numpy as np

from robo.task.rembo import REMBO


class Test(unittest.TestCase):

    def test_rembo(self):

        class TestTask(REMBO):

            def __init__(self):
                X_lower = np.array([-5, 0])
                X_upper = np.array([10, 15])
                super(TestTask, self).__init__(X_lower, X_upper, d=1)

            def objective_function(self, x):
                return x
        t = TestTask()

        x = np.random.uniform(low=-1.0, high=1.0, size=(100, 1))
        projected_scaled_x = t.evaluate(x)

        assert len(projected_scaled_x.shape) == 2
        assert projected_scaled_x.shape[1] == t.d_orig
        assert np.all(projected_scaled_x >= t.original_X_lower)
        assert np.all(projected_scaled_x <= t.original_X_upper)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_rembo']
    unittest.main()