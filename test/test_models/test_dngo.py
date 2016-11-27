import unittest
import numpy as np

from robo.models.dngo import DNGO


class TestDNGO(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 1)
        y = self.X * 2
        self.y = y[:, 0]
        self.model = DNGO()
        self.model.train(self.X, self.y, do_optimize=False)

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

    def test_get_incumbent(self):
        inc, inc_val = self.model.get_incumbent()

        b = np.argmin(self.y)
        np.testing.assert_almost_equal(inc, self.X[b], decimal=5)
        assert inc_val == self.y[b]

if __name__ == "__main__":
    unittest.main()
