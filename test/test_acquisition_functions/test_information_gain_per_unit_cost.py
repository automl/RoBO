import unittest
import george
import numpy as np

from robo.models.gaussian_process import GaussianProcess
from robo.priors import default_priors
from robo.acquisition_functions.information_gain_per_unit_cost import InformationGainPerUnitCost


class Test(unittest.TestCase):

    def setUp(self):

        self.X = np.random.rand(10, 2)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        self.c = np.exp(self.X[:, 1])
        self.n_dims = 2
        self.lower = np.zeros(self.n_dims)
        self.upper = np.ones(self.n_dims)
        self.is_env = np.array([0, 1])

        kernel = george.kernels.Matern52Kernel(np.array([0.1, 0.1]), ndim=2)
        self.model = GaussianProcess(kernel)
        self.model.train(self.X, self.y)

        kernel = george.kernels.Matern52Kernel(np.ones([self.n_dims]) * 0.01,
                                                       ndim=self.n_dims)

        kernel = 3000 * kernel

        prior = default_priors.TophatPrior(-2, 2)
        model = GaussianProcess(kernel, prior=prior)

        cost_kernel = george.kernels.Matern52Kernel(np.ones([self.n_dims]) * 0.01,
                                                       ndim=self.n_dims)

        cost_kernel = 3000 * cost_kernel

        prior = default_priors.TophatPrior(-2, 2)
        cost_model = GaussianProcess(cost_kernel, prior=prior)

        model.train(self.X, self.y, do_optimize=False)
        cost_model.train(self.X, self.c, do_optimize=False)
        self.acquisition_func = InformationGainPerUnitCost(model,
                                                           cost_model,
                                                           self.lower,
                                                           self.upper,
                                                           self.is_env)

        self.acquisition_func.update(model, cost_model)

    def test_sampling_representer_points(self):
        # Check if representer points are inside the configuration subspace
        assert np.any(self.acquisition_func.zb[:, self.is_env == 1] ==
                      self.acquisition_func.upper[self.is_env == 1])

    def test_compute(self):
        X_test = np.random.rand(5, 2)
        a = self.acquisition_func.compute(X_test, derivative=False)
        assert a.shape[0] == X_test.shape[0]
        assert len(a.shape) == 1

if __name__ == "__main__":
    unittest.main()
