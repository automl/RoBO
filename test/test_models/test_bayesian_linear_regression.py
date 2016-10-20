import unittest
import numpy as np

from robo.models.bayesian_linear_regression import BayesianLinearRegression
from robo.priors.dngo_priors import DNGOPrior


class TestBayesianLinearRegression(unittest.TestCase):

    def setUp(self):
        X = np.random.rand(10, 2)
        y = np.sinc(X * 10 - 5).sum(axis=1)

        prior = DNGOPrior()
        self.model = BayesianLinearRegression(prior=prior,
                                              burnin_steps=100,
                                              n_hypers=6,
                                              chain_length=200)
        self.model.train(X, y)

    def test_predict(self):
        X_test = np.random.rand(10, 2)

        m, v = self.model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]

    def test_marginal_log_likelihood(self):
        theta = np.array([0.2, 0.2, 0.001])
        mll = self.model.marginal_log_likelihood(theta)
        assert type(mll) == np.float

    def test_negative_mll(self):
        theta = np.array([0.2, 0.2, 0.001])
        mll = self.model.negative_mll(theta)
        assert type(mll) == np.float

    def test_posterior(self):
        pass
