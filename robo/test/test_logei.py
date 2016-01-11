import setup_logger
import unittest

import numpy as np

import GPy

from  scipy.optimize import check_grad

from robo.models.gpy_model import GPyModel
from robo.acquisition.log_ei import LogEI
from robo.initial_design.init_random_uniform import init_random_uniform


class LogEITestCase(unittest.TestCase):

    def setUp(self):
        self.X_lower = np.array([0])
        self.X_upper = np.array([1])
        self.X = np.random.rand(10)[:, np.newaxis]
        self.Y = np.sin(self.X)
        self.kernel = GPy.kern.RBF(input_dim=1)

        self.model = GPyModel(self.kernel)
        self.model.train(self.X, self.Y)
        self.log_ei = LogEI(self.model,
                    X_upper=self.X_upper,
                    X_lower=self.X_lower)

    def test_general_interface(self):

        X_test = init_random_uniform(self.X_lower, self.X_upper, 10)

#        a, dadx = self.log_ei(X_test, True)
        a = self.log_ei(X_test)

        assert len(a.shape) == 2
        assert a.shape[0] == X_test.shape[0]
        assert a.shape[1] == 1
        #assert len(dadx.shape) == 2
        #assert dadx.shape[0] == X_test.shape[0]
        #assert dadx.shape[1] == X_test.shape[1

#    def test_check_grads(self):
#        x_ = np.array([[np.random.rand()]])

#        assert check_grad(self.log_ei, lambda x: -self.ei(x, True)[1], x_) < 1e-5

if __name__ == "__main__":
    unittest.main()
