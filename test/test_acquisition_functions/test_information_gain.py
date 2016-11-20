import unittest
import numpy as np

from robo.acquisition_functions.information_gain import InformationGain

from test.dummy_model import DemoModel


class TestInformationGain(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        self.model = DemoModel()
        self.model.train(self.X, self.y)

    def test_compute(self):
        ig = InformationGain(self.model, np.zeros([2]), np.ones([2]))

        X_test = np.random.rand(5, 2)
        a = ig.compute(X_test, derivative=False)
        assert a.shape[0] == X_test.shape[0]
        assert len(a.shape) == 1

if __name__ == "__main__":
    unittest.main()
