import unittest
import numpy as np

from robo.util.posterior_optimization import posterior_mean_optimization, posterior_mean_plus_std_optimization

from test.dummy_model import DemoQuadraticModel


class TestPosteriorOptimization(unittest.TestCase):

    def setUp(self):
        X = np.random.randn(5, 2)
        y = np.sum((0.5 - X) ** 2, axis=1)

        self.model = DemoQuadraticModel()
        self.model.train(X, y)
        self.lower = np.array([0, 0])
        self.upper = np.array([1, 1])
        self.opt = np.array([0.5, 0.5])

    def test_posterior_mean_optimization(self):
        x = posterior_mean_optimization(self.model, self.lower, self.upper, method="cma", n_restarts=1)
        np.testing.assert_almost_equal(x, self.opt, decimal=5)

        x = posterior_mean_optimization(self.model, self.lower, self.upper, method="scipy", with_gradients=False)
        np.testing.assert_almost_equal(x, self.opt, decimal=5)

    def test_posterior_mean_plus_std_optimization(self):
        x = posterior_mean_plus_std_optimization(self.model, self.lower, self.upper, method="cma", n_restarts=1)
        np.testing.assert_almost_equal(x, self.opt, decimal=5)

        x = posterior_mean_optimization(self.model, self.lower, self.upper, method="scipy", with_gradients=False)
        np.testing.assert_almost_equal(x, self.opt, decimal=5)


if __name__ == "__main__":
    unittest.main()
