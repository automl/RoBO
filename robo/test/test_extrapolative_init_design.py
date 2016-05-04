'''
Created on Dec 21, 2015

@author: Aaron Klein
'''
import unittest
import numpy as np

from robo.initial_design.extrapolative_initial_design import extrapolative_initial_design


class TestEnvInitDesign(unittest.TestCase):

    def test(self):
        l = np.array([0, 0])
        u = np.array([1, 1])
        is_env = np.array([0, 1])
        N = 100
        X = extrapolative_initial_design(l, u, is_env, N)

        assert len(X.shape) == 2
        assert X.shape[0] == N
        assert X.shape[1] == 2
        for i in range(N):
            assert X[i, 1] == 1 / float(2 ** (i % 4 + 2))

if __name__ == "__main__":
    unittest.main()
