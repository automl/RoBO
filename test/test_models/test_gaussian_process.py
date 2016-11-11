import george
import unittest
import numpy as np

from robo.models.gaussian_process import GaussianProcess
from robo.priors.default_priors import TophatPrior


class TestGaussianProcess(unittest.TestCase):

    def setUp(self):
        X = np.random.rand(10, 2)
        y = np.sinc(X * 10 - 5).sum(axis=1)

        kernel = george.kernels.Matern52Kernel(np.ones(X.shape[1]),
                                               ndim=X.shape[1])

        prior = TophatPrior(-2, 2)
        self.model = GaussianProcess(kernel, prior=prior)
        self.model.train(X, y, do_optimize=False)

    def test_predict(self):
        X_test = np.random.rand(10, 2)

        m, v = self.model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]

        m, v = self.model.predict(X_test, full_cov=True)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 2
        assert v.shape[0] == X_test.shape[0]
        assert v.shape[1] == X_test.shape[0]

    def test_sample_function(self):
        X_test = np.random.rand(8, 2)
        n_funcs = 3
        funcs = self.model.sample_functions(X_test, n_funcs=n_funcs)

        assert len(funcs.shape) == 2
        assert funcs.shape[0] == n_funcs
        assert funcs.shape[1] == X_test.shape[0]

    def test_predict_variance(self):
        x_test1 = np.random.rand(1, 2)
        x_test2 = np.random.rand(10, 2)
        var = self.model.predict_variance(x_test1, x_test2)
        assert len(var.shape) == 2
        assert var.shape[0] == x_test2.shape[0]
        assert var.shape[1] == x_test1.shape[0]

    def test_nll(self):
        theta = np.array([0.2, 0.2, 0.001])
        nll = self.model.nll(theta)

    def test_optimize(self):
        theta = self.model.optimize()
        # Hyperparameters are 2 length scales + noise
        assert theta.shape[0] == 3


if __name__ == "__main__":
    unittest.main()
