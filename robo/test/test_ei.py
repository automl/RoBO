import sys
import os
#sys.path.insert(0, '../')
import unittest
import errno
import numpy as np
import random
import GPy
from robo.models.GPyModel import GPyModel
from robo.acquisition.EI import EI
from robo.recommendation.incumbent import compute_incumbent

here = os.path.abspath(os.path.dirname(__file__))


class EITestCase1(unittest.TestCase):

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

        best = np.argmin(self.y)
        incumbent = self.x[best]

        ei_par = EI(self.model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.0)

        out0 = ei_par(incumbent[:, np.newaxis], derivative=True)
        value0 = out0[0]
        derivative0 = out0[1]
        assert(value0[0] <= 1e-5)

        x_value = incumbent + np.random.random_integers(1, 10) / 1000.
        out1 = ei_par(x_value[:, np.newaxis], derivative=True)
        value1 = out1[0]
        derivative1 = out1[1]

        assert(np.all(value0 < value1))
        assert(np.all(np.abs(derivative0) < np.abs(derivative1)))


if __name__ == "__main__":
    unittest.main()
