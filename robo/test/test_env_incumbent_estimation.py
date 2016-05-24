'''
Created on Dec 18, 2015

@author: Aaron Klein
'''
import unittest
import george
import numpy as np


from robo.models.gaussian_process import GaussianProcess
from robo.priors.default_priors import TophatPrior
from robo.initial_design.init_random_uniform import init_random_uniform

from robo.incumbent.env_posterior_opt import EnvPosteriorMeanOptimization
from robo.incumbent.env_posterior_opt import EnvPosteriorMeanAndStdOptimization


class TestEnvPosteriorMeanOptimization(unittest.TestCase):

    def test_scipy(self):
        X_lower = np.array([0, 0])
        X_upper = np.array([1, 1])
        is_env = np.array([0, 1])

        X = init_random_uniform(X_lower, X_upper, 10)
        Y = np.sin(X[:, 0])[:, np.newaxis]

        kernel = george.kernels.Matern52Kernel(np.ones([1, 1]),
                                               ndim=2)

        prior = TophatPrior(-2, 2)
        model = GaussianProcess(kernel, prior=prior)
        model.train(X, Y)

        rec = EnvPosteriorMeanOptimization(model, X_lower, X_upper, is_env)
        startpoints = init_random_uniform(X_lower, X_upper, 10)
        inc, inc_val = rec.estimate_incumbent(startpoints)

        # Check shapes
        assert len(inc.shape) == 2
        assert inc.shape[0] == 1
        assert inc.shape[1] == X_lower.shape[0]

        assert len(inc_val.shape) == 2
        assert inc_val.shape[0] == 1
        assert inc_val.shape[1] == 1

        # Check if incumbent is in the bounds
        assert not np.any([np.any(inc[:, i] < X_lower[i])
                        for i in range(X_lower.shape[0])])
        assert not np.any([np.any(inc[:, i] > X_upper[i])
                        for i in range(X_upper.shape[0])])

        # Check if incumbent is in the projected subspace
        assert inc[:, is_env == 1] == X_upper[is_env == 1]

    def test_cmaes(self):
        X_lower = np.array([0, 0, 0])
        X_upper = np.array([1, 1, 1])
        is_env = np.array([0, 0, 1])

        X = init_random_uniform(X_lower, X_upper, 10)
        Y = np.sin(X[:, 0])[:, np.newaxis]

        kernel = george.kernels.Matern52Kernel(np.ones([1, 1]),
                                               ndim=3)

        prior = TophatPrior(-2, 2)
        model = GaussianProcess(kernel, prior=prior)
        model.train(X, Y)

        rec = EnvPosteriorMeanOptimization(model, X_lower, X_upper, is_env,
                                           method="cmaes")
        startpoints = init_random_uniform(X_lower, X_upper, 10)
        inc, inc_val = rec.estimate_incumbent(startpoints)

        # Check shapes
        assert len(inc.shape) == 2
        assert inc.shape[0] == 1
        assert inc.shape[1] == X_lower.shape[0]

        assert len(inc_val.shape) == 2
        assert inc_val.shape[0] == 1
        assert inc_val.shape[1] == 1

        # Check if incumbent is in the bounds
        assert not np.any([np.any(inc[:, i] < X_lower[i])
                        for i in range(X_lower.shape[0])])
        assert not np.any([np.any(inc[:, i] > X_upper[i])
                        for i in range(X_upper.shape[0])])

        # Check if incumbent is in the projected subspace
        assert inc[:, is_env == 1] == X_upper[is_env == 1]


class TestEnvPosteriorMeanAndStdOptimization(unittest.TestCase):

    def test_scipy(self):
        X_lower = np.array([0, 0])
        X_upper = np.array([1, 1])
        is_env = np.array([0, 1])

        X = init_random_uniform(X_lower, X_upper, 10)
        Y = np.sin(X[:, 0])[:, np.newaxis]

        kernel = george.kernels.Matern52Kernel(np.ones([1, 1]),
                                               ndim=2)

        prior = TophatPrior(-2, 2)
        model = GaussianProcess(kernel, prior=prior)
        model.train(X, Y)

        rec = EnvPosteriorMeanAndStdOptimization(model, X_lower,
                                                 X_upper, is_env)
        startpoints = init_random_uniform(X_lower, X_upper, 10)
        inc, inc_val = rec.estimate_incumbent(startpoints)

        # Check shapes
        assert len(inc.shape) == 2
        assert inc.shape[0] == 1
        assert inc.shape[1] == X_lower.shape[0]

        assert len(inc_val.shape) == 2
        assert inc_val.shape[0] == 1
        assert inc_val.shape[1] == 1

        # Check if incumbent is in the bounds
        assert not np.any([np.any(inc[:, i] < X_lower[i])
                        for i in range(X_lower.shape[0])])
        assert not np.any([np.any(inc[:, i] > X_upper[i])
                        for i in range(X_upper.shape[0])])

        # Check if incumbent is in the projected subspace
        assert inc[:, is_env == 1] == X_upper[is_env == 1]

    def test_cmaes(self):
        X_lower = np.array([0, 0, 0])
        X_upper = np.array([1, 1, 1])
        is_env = np.array([0, 0, 1])

        X = init_random_uniform(X_lower, X_upper, 10)
        Y = np.sin(X[:, 0])[:, np.newaxis]

        kernel = george.kernels.Matern52Kernel(np.ones([1, 1]),
                                               ndim=3)

        prior = TophatPrior(-2, 2)
        model = GaussianProcess(kernel, prior=prior)
        model.train(X, Y)

        rec = EnvPosteriorMeanAndStdOptimization(model, X_lower, X_upper,
                                                 is_env,
                                                 method="cmaes")
        startpoints = init_random_uniform(X_lower, X_upper, 10)
        inc, inc_val = rec.estimate_incumbent(startpoints)

        print(inc, inc_val)
        # Check shapes
        assert len(inc.shape) == 2
        assert inc.shape[0] == 1
        assert inc.shape[1] == X_lower.shape[0]

        assert len(inc_val.shape) == 2
        assert inc_val.shape[0] == 1
        assert inc_val.shape[1] == 1

        # Check if incumbent is in the bounds
        assert not np.any([np.any(inc[:, i] < X_lower[i])
                        for i in range(X_lower.shape[0])])
        assert not np.any([np.any(inc[:, i] > X_upper[i])
                        for i in range(X_upper.shape[0])])

        # Check if incumbent is in the projected subspace
        assert inc[:, is_env == 1] == X_upper[is_env == 1]

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
