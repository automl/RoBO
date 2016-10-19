import unittest
import numpy as np

from robo.util import normalization


class TestNormalization(unittest.TestCase):

    def test_zero_one_normalization(self):

        X = np.random.randn(100, 3)
        X_norm, lo, up = normalization.zero_one_normalization(X)

        assert X_norm.shape == X.shape
        assert np.min(X_norm) >= 0
        assert np.max(X_norm) <= 1
        assert lo.shape[0] == X.shape[1]
        assert up.shape[0] == X.shape[1]

    def test_zero_one_unnormalization(self):
        X_norm = np.random.rand(100, 3)
        lo = np.ones([3]) * -1
        up = np.ones([3])
        X = normalization.zero_one_unnormalization(X_norm, lo, up)

        assert X_norm.shape == X.shape
        assert np.all(np.min(X, axis=0) >= lo)
        assert np.all(np.max(X, axis=0) <= up)

    def test_zero_mean_unit_var_normalization(self):
        X = np.random.rand(100, 3)
        X_norm, m, s = normalization.zero_mean_unit_var_normalization(X)

        np.testing.assert_almost_equal(np.mean(X_norm, axis=0), np.zeros(X_norm.shape[1]), decimal=1)
        np.testing.assert_almost_equal(np.var(X_norm, axis=0), np.ones(X_norm.shape[1]), decimal=1)

        assert X_norm.shape == X.shape
        assert m.shape[0] == X.shape[1]
        assert s.shape[0] == X.shape[1]

    def test_zero_one_unit_var_unnormalization(self):
        X_norm = np.random.randn(100, 3)
        m = np.ones(X_norm.shape[1]) * 3
        s = np.ones(X_norm.shape[1]) * 0.1
        X = normalization.zero_mean_unit_var_unnormalization(X_norm, m, s)

        np.testing.assert_almost_equal(np.mean(X, axis=0), m, decimal=1)
        np.testing.assert_almost_equal(np.var(X, axis=0), s**2, decimal=1)

        assert X_norm.shape == X.shape

if __name__ == "__main__":
    unittest.main()
