import sys
import os
#sys.path.insert(0, '../')
import unittest
import errno
import numpy as np
import random
import GPy
from robo.models.GPyModel import GPyModel
from robo.acquisition.LogEI import LogEI


here = os.path.abspath(os.path.dirname(__file__))

class LogEITestCase1(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[ 0.62971589], [ 0.63273273], [ 0.17867868], [ 0.17447447], [ 1.88558559]]);
        self.y = np.array([[-3.69925653], [-3.66221988], [-3.65560591], [-3.58907791], [-8.06925984]]);
        self.kernel = GPy.kern.RBF(input_dim=1, variance= 30.1646253727, lengthscale = 0.435343653946)    
        self.noise = 1e-10
        self.model = GPyModel(self.kernel, noise_variance=self.noise, optimize=False)
        self.model.train(self.x, self.y)
        self.bigger_noise = 1e2
        self.noise_model = GPyModel(self.kernel, noise_variance=self.bigger_noise, optimize=False)
        self.noise_model.train(self.x, self.y)

    def test(self):
        X_upper = np.array([ 2.1])
        X_lower = np.array([-2.1])
        log_ei_par_0 = LogEI(self.model, X_upper=X_upper, X_lower=X_lower,  par=0.0, derivative=True)
        log_ei_par_1 = LogEI(self.model, X_upper=X_upper, X_lower=X_lower,  par=1.0, derivative=True)
        log_ei_par_2 = LogEI(self.model, X_upper=X_upper, X_lower=X_lower,  par=2.0, derivative=True)
        x_values = [0.62971589] + [2.1 * random.random() - 2.1 for i in range(10)]
        value0 = np.array([ log_ei_par_0(np.array([[x]])) for x in x_values])
        value1 = np.array([ log_ei_par_1(np.array([[x]])) for x in x_values])
        value2 = np.array([ log_ei_par_2(np.array([[x]])) for x in x_values])
        
        assert(value0[0] <= 1e-5)
        assert(np.all(np.logical_or(value0 >= value1, np.isinf(value0), np.isinf(value1))))
        assert(np.all(value1 >= value2))
        log_ei_par_0.update(self.noise_model)
        value0_noise = np.array([ log_ei_par_0(np.array([[x]])) for x in x_values])
        assert(np.all(np.logical_or(value0_noise >= value0, np.isinf(value0), np.isinf(value0_noise))))


if __name__=="__main__":
    unittest.main()
