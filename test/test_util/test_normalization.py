import unittest
import numpy as np

from robo.util import normalization


class TestNormalization(unittest.TestCase):

    def test_zero_one_normalization(self):

        X = np.random.randn(10, 3)
        X_norm, lo, up = normalization.zero_one_normalization(X)

        assert X_norm.shape == X.shape
        assert np.min(X_norm) >= 0
        assert np.max(X_norm) <= 1
        assert lo.shape[0] == X.shape[1]
        assert up.shape[0] == X.shape[1]

    def test_zero_one_unnormalization(self):
        X_norm = np.random.rand(10, 3)
        lo = np.ones([3]) * -1
        up = np.ones([3])
        X = normalization.zero_mean_unnormalization(X_norm, lo, up)

        assert X_norm.shape == X.shape
        assert np.all(np.min(X, axis=0) >= lo)
        assert np.all(np.max(X, axis=0) <= up)
        assert lo.shape[0] == X_norm.shape[1]
        assert up.shape[0] == X_norm.shape[1]

    def test_zero_mean_normalization(self):
        X = np.random.rand(10, 3)
        X_norm, m, s = normalization.zero_mean_normalization(X)

        assert X_norm.shape == X.shape
        unittest.assertNotAlmostEquals(np.mean(X_norm, axis=0), m)
        unittest.assertNotAlmostEquals(np.std(X_norm, axis=0), s)
        assert m.shape[0] == X.shape[1]
        assert s.shape[0] == X.shape[1]

#
# def test_zero_one_unnormalization(self):
#     X_norm = np.random.rand(10, 3)
#     lo = np.ones([3]) * -1
#     up = np.ones([3])
#     X = normalization.zero_mean_unnormalization(X_norm, lo, up)
#
#     assert X_norm.shape == X.shape
#     assert np.all(np.min(X, axis=0) >= lo)
#     assert np.all(np.max(X, axis=0) <= up)
#     assert lo.shape[0] == X_norm.shape[1]
#     assert up.shape[0] == X_norm.shape[1]


if __name__ == "__main__":
    unittest.main()
