import unittest
import numpy as np

from robo.acquisition_functions.pi import PI

from test.dummy_model import DemoModel


class TestPI(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        self.model = DemoModel()
        self.model.train(self.X, self.y)

    def test_compute(self):
        pi = PI(self.model)

        X_test = np.random.rand(5, 2)
        a = pi.compute(X_test, derivative=False)
        assert a.shape[0] == X_test.shape[0]
        assert len(a.shape) == 1

if __name__ == "__main__":
    unittest.main()
