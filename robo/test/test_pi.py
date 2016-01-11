
import logging
import unittest
import numpy as np

from scipy.optimize import check_grad

import GPy

from robo.models.gpy_model import GPyModel
from robo.acquisition.pi import PI
from robo.initial_design.init_random_uniform import init_random_uniform

logger = logging.getLogger(__name__)


class PITestCase(unittest.TestCase):

    def setUp(self):
        self.X_lower = np.array([0])
        self.X_upper = np.array([1])
        self.X = init_random_uniform(self.X_lower, self.X_upper, 10)
        self.Y = np.sin(self.X)
        self.kernel = GPy.kern.RBF(input_dim=1)

        self.model = GPyModel(self.kernel)
        self.model.train(self.X, self.Y)
        self.pi = PI(self.model,
                    X_upper=self.X_upper,
                    X_lower=self.X_lower)

    def test_general_interface(self):

        X_test = init_random_uniform(self.X_lower, self.X_upper, 10)
        # Just check if PI is always greater equal than 0

        a, dadx = self.pi(X_test, True)

        assert len(a.shape) == 2
        assert a.shape[0] == X_test.shape[0]
        assert a.shape[1] == 1
        assert np.all(a) >= 0.0
        assert len(dadx.shape) == 2
        assert dadx.shape[0] == X_test.shape[0]
        assert dadx.shape[1] == X_test.shape[1]

    def test_check_grads(self):
        x_ = np.array([[np.random.rand()]])

        assert check_grad(self.pi, lambda x: -self.pi(x, True)[1], x_) < 1e-5

if __name__ == "__main__":
    unittest.main()
