import george
import unittest
import numpy as np

from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.priors.default_priors import TophatPrior


class TestGaussianProcessMCMC(unittest.TestCase):

    def setUp(self):
        self.X = np.random.randn(10, 2)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)

        kernel = george.kernels.Matern52Kernel(np.ones(self.X.shape[1]),
                                               ndim=self.X.shape[1])

        prior = TophatPrior(-2, 2)
        self.model = GaussianProcessMCMC(kernel,
                                         prior=prior,
                                         n_hypers=6,
                                         burnin_steps=100,
                                         chain_length=200)
        self.model.train(self.X, self.y, do_optimize=True)

    def test_predict(self):
        X_test = np.random.rand(10, 2)

        m, v = self.model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]

    def test_loglikelihood(self):
        theta = np.array([0.2, 0.2, 0.001])
        ll = self.model.loglikelihood(theta)

    def test_get_incumbent(self):
        inc, inc_val = self.model.get_incumbent()

        b = np.argmin(self.y)

        np.testing.assert_almost_equal(inc, self.X[b], decimal=5)
        assert inc_val == self.y[b]

if __name__ == "__main__":
    unittest.main()