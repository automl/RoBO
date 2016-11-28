import unittest
import numpy as np

from robo.acquisition_functions.lcb import LCB

from test.dummy_model import DemoModel


class TestLCB(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        self.model = DemoModel()
        self.model.train(self.X, self.y)

    def test_compute(self):
        lcb = LCB(self.model)

        X_test = np.random.rand(5, 2)
        a = lcb.compute(X_test, derivative=False)
        assert a.shape[0] == X_test.shape[0]
        assert len(a.shape) == 1

        np.testing.assert_almost_equal(a, np.ones(X_test.shape[0]) * (- np.mean(self.y) + np.std(self.y)), decimal=3)

if __name__ == "__main__":
    unittest.main()
