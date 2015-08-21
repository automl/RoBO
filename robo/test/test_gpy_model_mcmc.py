'''
Created on Jun 25, 2015

@author: Aaron Klein
'''
import unittest

import GPy
import numpy as np

from robo.models.gpy_model_mcmc import GPyModelMCMC


class TestGPyModelMCMC(unittest.TestCase):

    def setUp(self):

        self.n_hypers = 10
        self.num_points = np.random.randint(1, 100)
        self.num_features = np.random.randint(1, 10)

        self.X = np.random.rand(self.num_points, self.num_features)
        self.y = np.random.rand(self.num_points)[:, np.newaxis]

        self.kernel = GPy.kern.Matern52(input_dim=self.num_features)
        self.model = GPyModelMCMC(self.kernel, burnin=10, chain_length=10, n_hypers=self.n_hypers)

    def test_train(self):
        self.model.train(self.X, self.y)
        mean, var = self.model.predict(self.X[0, np.newaxis])

        assert mean.shape[0] == self.n_hypers
        assert var.shape[0] == self.n_hypers

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
