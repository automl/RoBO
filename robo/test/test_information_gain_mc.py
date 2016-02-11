# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:33:59 2015

@author: aaron
"""

import unittest
import george
import numpy as np

from robo.models.gaussian_process import GaussianProcess
from robo.priors import default_priors
from robo.initial_design.init_random_uniform import init_random_uniform

from robo.acquisition.information_gain_mc import InformationGainMC
from enves.sin_func import SinFunction


class InformationGainMCTestCase(unittest.TestCase):

    def setUp(self):
        self.task = SinFunction()

        kernel = george.kernels.Matern52Kernel(np.ones([self.task.n_dims]) * 0.01,
                                                       ndim=self.task.n_dims)

        noise_kernel = george.kernels.WhiteKernel(1e-9, ndim=self.task.n_dims)
        kernel = 3000 * (kernel + noise_kernel)

        prior = default_priors.TophatPrior(-2, 2)
        model = GaussianProcess(kernel, prior=prior)
        X = init_random_uniform(self.task.X_lower, self.task.X_upper, 3)
        Y = self.task.evaluate(X)

        model.train(X, Y, do_optimize=False)
        self.acquisition_func = InformationGainMC(model,
                     X_upper=self.task.X_upper,
                     X_lower=self.task.X_lower)

        self.acquisition_func.update(model)

    def test_sampling_representer_points(self):

        # Check if representer points are inside the bounds
        assert np.any(self.acquisition_func.zb >= self.acquisition_func.X_lower)
        assert np.any(self.acquisition_func.zb <= self.acquisition_func.X_upper)

    def test_compute_pmin(self):

        # Uniform distribution
        m = np.ones([self.acquisition_func.Nb, 1])
        v = np.eye(self.acquisition_func.Nb)

        pmin = self.acquisition_func.compute_pmin(m, v)
        uprob = 1. / self.acquisition_func.Nb

        assert pmin.shape[0] == self.acquisition_func.Nb
        assert np.any(pmin < (uprob + 0.03)) and np.any(pmin > uprob - 0.01)

        # Dirac delta
        m = np.ones([self.acquisition_func.Nb, 1]) * 1000
        m[0] = 1
        v = np.eye(self.acquisition_func.Nb)

        pmin = self.acquisition_func.compute_pmin(m, v)
        uprob = 1. / self.acquisition_func.Nb
        assert pmin[0] == 1.0
        assert np.any(pmin[:1] > 1e-10)

        # Check uniform case with halluzinated values
        m = np.ones([self.acquisition_func.Nb, 50]) * 1000
        for i in range(50):
            m[i, i] = 1

        v = np.eye(self.acquisition_func.Nb)

        pmin = self.acquisition_func.compute_pmin(m, v)

        assert pmin.shape[0] == self.acquisition_func.Nb
        assert np.any(pmin < (uprob + 0.03)) and np.any(pmin > uprob - 0.01)

    def test_innovations(self):
        # Case 1: Assume no influence of test point on representer points
        rep = np.array([[1.0]])
        x = np.array([[0.0]])
        dm, dv = self.acquisition_func.innovations(x, rep)
        assert np.any(np.abs(dm) < 1e-4)
        assert np.any(np.abs(dv) < 1e-4)

        # Case 2: Test point is close to representer points
        rep = np.array([[1.0]])
        x = np.array([[0.99]])
        dm, dv = self.acquisition_func.innovations(x, rep)
        assert np.any(np.abs(dm) > 1e-4)
        assert np.any(np.abs(dv) > 1e-4)

    def test_general_interface(self):

        X_test = init_random_uniform(self.task.X_lower, self.task.X_upper, 1)

        a = self.acquisition_func(X_test, False)

        assert len(a.shape) == 2
        assert a.shape[0] == X_test.shape[0]
        assert a.shape[1] == 1

if __name__ == "__main__":
    unittest.main()
