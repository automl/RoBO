'''
Created on Dec 18, 2015

@author: Aaron Klein
'''
import unittest
import george
import numpy as np

from robo.models.gaussian_process import GaussianProcess
from robo.priors import default_priors
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.acquisition.information_gain_per_unit_cost import InformationGainPerUnitCost
from robo.task.synthetic_functions.sin_func import SinFunction
from robo.task.environmental_synthetic_function import SyntheticFunctionWrapper


class Test(unittest.TestCase):

    def setUp(self):
        s = SinFunction()
        self.task = SyntheticFunctionWrapper(s)

        kernel = george.kernels.Matern52Kernel(np.ones([self.task.n_dims]) * 0.01,
                                                       ndim=self.task.n_dims)

        noise_kernel = george.kernels.WhiteKernel(1e-9, ndim=self.task.n_dims)
        kernel = 3000 * (kernel + noise_kernel)

        prior = default_priors.TophatPrior(-2, 2)
        model = GaussianProcess(kernel, prior=prior)

        cost_kernel = george.kernels.Matern52Kernel(np.ones([self.task.n_dims]) * 0.01,
                                                       ndim=self.task.n_dims)

        cost_noise_kernel = george.kernels.WhiteKernel(1e-9,
                                                       ndim=self.task.n_dims)
        cost_kernel = 3000 * (cost_kernel + cost_noise_kernel)

        prior = default_priors.TophatPrior(-2, 2)
        cost_model = GaussianProcess(cost_kernel, prior=prior)

        X = init_random_uniform(self.task.X_lower, self.task.X_upper, 3)
        Y, C = self.task.evaluate(X)

        model.train(X, Y, do_optimize=False)
        cost_model.train(X, C, do_optimize=False)
        self.acquisition_func = InformationGainPerUnitCost(model,
                                                    cost_model,
                                                    self.task.X_lower,
                                                    self.task.X_upper,
                                                    self.task.is_env)

        self.acquisition_func.update(model, cost_model)

    def test_sampling_representer_points(self):
        # Check if representer points are inside the configuration subspace
        assert np.any(self.acquisition_func.zb[self.task.is_env == 1] ==
                        self.acquisition_func.X_upper[self.task.is_env == 1])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
