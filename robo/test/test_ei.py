import setup_logger

import GPy
import unittest
import numpy as np
from  scipy.optimize import check_grad

from robo.models.gpy_model import GPyModel
from robo.acquisition.ei import EI
from robo.recommendation.incumbent import compute_incumbent


class EITestCase(unittest.TestCase):

    def setUp(self):
        self.X_lower = np.array([0])
        self.X_upper = np.array([1])
        self.X = np.random.rand(10)[:, np.newaxis]
        self.Y = np.sin(self.X)
        self.kernel = GPy.kern.RBF(input_dim=1)

        self.model = GPyModel(self.kernel)
        self.model.train(self.X, self.Y)
        self.ei = EI(self.model,
                    X_upper=self.X_upper,
                    X_lower=self.X_lower,
                    compute_incumbent=compute_incumbent)

    def test_general_interface(self):

        X_test = np.random.rand(1000)[:, np.newaxis]
        # Just check if EI is always greater equal than 0
        for x in X_test:
            assert self.ei(x[np.newaxis, :]) >= 0.0

    def test_check_grads(self):
        x_ = np.array([[np.random.rand()]])
        print x_
        print self.ei(x_, True)
        assert check_grad(self.ei, lambda x: -self.ei(x, True)[1], x_) < 1e-5


if __name__ == "__main__":
    unittest.main()
