import setup_logger
import unittest

import numpy as np

import GPy
from robo.models.gpy_model import GPyModel
from robo.acquisition.LogEI import LogEI
from robo.recommendation.incumbent import compute_incumbent


class LogEITestCase1(unittest.TestCase):

    def setUp(self):
        self.x = np.array([[0.62971589], [0.63273273], [0.17867868], [0.17447447], [1.88558559]])
        self.y = np.array([[-3.69925653], [-3.66221988], [-3.65560591], [-3.58907791], [-8.06925984]])
        self.kernel = GPy.kern.RBF(input_dim=1, variance=30.1646253727, lengthscale=0.435343653946)
        self.noise = 1e-20
        self.model = GPyModel(self.kernel, noise_variance=self.noise, optimize=False)
        self.model.train(self.x, self.y)

    def test(self):
        X_upper = np.array([2.1])
        X_lower = np.array([-2.1])

        x_test = np.array([[1.7], [2.0]])
        log_ei_estimator = LogEI(self.model, X_lower, X_upper, compute_incumbent=compute_incumbent)

        assert log_ei_estimator(x_test[0, np.newaxis])[0] > -np.Infinity
        assert log_ei_estimator(x_test[1, np.newaxis])[0] > -np.Infinity

        assert(log_ei_estimator(self.x[-1, np.newaxis])[0]) == -np.Infinity


if __name__ == "__main__":
    unittest.main()
