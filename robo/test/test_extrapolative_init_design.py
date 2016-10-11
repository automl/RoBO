'''
Created on Dec 21, 2015

@author: Aaron Klein
'''
import unittest
import numpy as np

from robo.task.base_task import BaseTask
from robo.initial_design.extrapolative_initial_design import extrapolative_initial_design


class DemoTask(BaseTask):

    def __init__(self):
        X_lower = np.array([0, np.log(10)])
        X_upper = np.array([1, np.log(1000)])
        self.is_env = np.array([0, 1])

        super(DemoTask, self).__init__(X_lower, X_upper)


class TestEnvInitDesign(unittest.TestCase):

    def test(self):
        task = DemoTask()
        N = 100
        X = extrapolative_initial_design(task, N)

        assert len(X.shape) == 2
        assert X.shape[0] == N
        assert X.shape[1] == 2
        for i in range(N):
            s = np.exp(task.retransform(X[i])[-1])
            assert np.round(s, 0) == np.round(1000. / float(2 ** (i % 4 + 2)), 0)

if __name__ == "__main__":
    unittest.main()
