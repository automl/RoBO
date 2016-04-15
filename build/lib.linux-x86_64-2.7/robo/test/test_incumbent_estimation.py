# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:51:37 2015

@author: aaron
"""

import setup_logger

import george
import unittest
import numpy as np


from robo.models.gaussian_process import GaussianProcess
from robo.priors.default_priors import TophatPrior
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.incumbent.posterior_optimization import PosteriorMeanOptimization
from robo.incumbent.posterior_optimization import PosteriorMeanAndStdOptimization
from robo.incumbent.best_observation import BestObservation


class TestBestObservation(unittest.TestCase):

    def test(self):
        X_lower = np.array([0])
        X_upper = np.array([1])
        X = init_random_uniform(X_lower, X_upper, 10)
        Y = np.sin(X)

        kernel = george.kernels.Matern52Kernel(np.ones([1]),
                                               ndim=1)

        prior = TophatPrior(-2, 2)
        model = GaussianProcess(kernel, prior=prior)
        model.train(X, Y)

        rec = BestObservation(model, X_lower, X_upper)
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


class TestPosteriorMeanOptimization(unittest.TestCase):

    def test_scipy(self):
        X_lower = np.array([0])
        X_upper = np.array([1])
        X = init_random_uniform(X_lower, X_upper, 10)
        Y = np.sin(X)

        kernel = george.kernels.Matern52Kernel(np.ones([1]),
                                               ndim=1)

        prior = TophatPrior(-2, 2)
        model = GaussianProcess(kernel, prior=prior)
        model.train(X, Y)

        rec = PosteriorMeanOptimization(model, X_lower, X_upper)
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

    def test_cmaes(self):
        X_lower = np.array([0, 0])
        X_upper = np.array([1, 1])
        X = init_random_uniform(X_lower, X_upper, 10)
        Y = np.sin(X[:, 0])[:, np.newaxis]

        kernel = george.kernels.Matern52Kernel(np.ones([1, 1]),
                                               ndim=2)

        prior = TophatPrior(-2, 2)
        model = GaussianProcess(kernel, prior=prior)
        model.train(X, Y)

        rec = PosteriorMeanOptimization(model, X_lower, X_upper,
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


class TestPosteriorMeanAndStdOptimization(unittest.TestCase):

    def test_scipy(self):
        X_lower = np.array([0])
        X_upper = np.array([1])
        X = init_random_uniform(X_lower, X_upper, 10)
        Y = np.sin(X)

        kernel = george.kernels.Matern52Kernel(np.ones([1]),
                                               ndim=1)

        prior = TophatPrior(-2, 2)
        model = GaussianProcess(kernel, prior=prior)
        model.train(X, Y)

        rec = PosteriorMeanAndStdOptimization(model, X_lower, X_upper)
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

    def test_cmaes(self):
        X_lower = np.array([0, 0])
        X_upper = np.array([1, 1])
        X = init_random_uniform(X_lower, X_upper, 10)
        Y = np.sin(X[:, 0])[:, np.newaxis]

        kernel = george.kernels.Matern52Kernel(np.ones([1, 1]),
                                               ndim=2)

        prior = TophatPrior(-2, 2)
        model = GaussianProcess(kernel, prior=prior)
        model.train(X, Y)

        rec = PosteriorMeanAndStdOptimization(model, X_lower, X_upper,
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


if __name__ == "__main__":
    unittest.main()
