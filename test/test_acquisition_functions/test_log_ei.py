import unittest
import numpy as np

from robo.acquisition_functions.log_ei import LogEI

from test.dummy_model import DemoModel


class TestLogEI(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        self.model = DemoModel()
        self.model.train(self.X, self.y)

    def test_compute(self):
        log_ei = LogEI(self.model)

        X_test = np.random.rand(5, 2)
        a = log_ei.compute(X_test, derivative=False)
        assert a.shape[0] == X_test.shape[0]
        assert len(a.shape) == 1

if __name__ == "__main__":
    unittest.main()
