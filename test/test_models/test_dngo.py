import unittest
import numpy as np

from robo.models.dngo import DNGO


class TestDNGO(unittest.TestCase):

    def setUp(self):
        X = np.random.rand(10, 1)
        y = X * 2
        y = y[:, 0]
        self.model = DNGO()
        self.model.train(X, y, do_optimize=False)

    def test_predict(self):
        X_test = np.random.rand(10, 1)

        m, v = self.model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]

    def test_marginal_log_likelihood(self):
        theta = np.array([np.log(1), np.log(1000)])
        mll = self.model.marginal_log_likelihood(theta)

    def test_negative_mll(self):
        theta = np.array([np.log(1), np.log(1000)])
        mll = self.model.negative_mll(theta)
