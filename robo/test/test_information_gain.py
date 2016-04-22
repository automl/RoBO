# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:33:59 2015

@author: aaron
"""

import george
import unittest
import numpy as np
from  scipy.optimize import check_grad

from robo.models.gaussian_process import GaussianProcess
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.priors import default_priors

from robo.acquisition.information_gain import InformationGain
from robo.task.synthetic_functions.sin_func import SinFunction
from robo.util import epmgp


class InformationGainTestCase(unittest.TestCase):

    def setUp(self):
        self.task = SinFunction()
        
        kernel = george.kernels.Matern52Kernel(np.ones([self.task.n_dims]) * 0.01,
                                                       ndim=self.task.n_dims)

        noise_kernel = george.kernels.WhiteKernel(1e-9, ndim=self.task.n_dims)
        kernel = 3000 * (kernel + noise_kernel)

        prior = default_priors.TophatPrior(-2, 2)
        model = GaussianProcess(kernel, prior=prior)
        X = init_random_uniform(self.task.X_lower,
                                self.task.X_upper,
                                3)
        Y = self.task.evaluate(X)

        model.train(X, Y, do_optimize=False)
        self.acquisition_func = InformationGain(model,
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

        pmin = epmgp.joint_min(m, v)
        pmin = np.exp(pmin)

        uprob = 1. / self.acquisition_func.Nb

        assert pmin.shape[0] == self.acquisition_func.Nb
        assert np.any(pmin < (uprob + 0.03)) and np.any(pmin > uprob - 0.01)

        # Dirac delta
        m = np.ones([self.acquisition_func.Nb, 1]) * 1000
        m[0] = 1
        v = np.eye(self.acquisition_func.Nb)

        pmin = epmgp.joint_min(m, v)
        pmin = np.exp(pmin)
        uprob = 1. / self.acquisition_func.Nb
        assert pmin[0] == 1.0
        assert np.any(pmin[:1] > 1e-10)

    def test_innovations(self):
        # Case 1: Assume no influence of test point on representer points
        rep = np.array([[1.0]])
        x = np.array([[0.0]])
        dm, dv = self.acquisition_func.innovations(x, rep)

        assert np.any(np.abs(dm) < 1e-3)
        assert np.any(np.abs(dv) < 1e-3)

        # Case 2: Test point is close to representer points
        rep = np.array([[1.0]])
        x = np.array([[0.99]])
        dm, dv = self.acquisition_func.innovations(x, rep)
        assert np.any(np.abs(dm) > 1e-3)
        assert np.any(np.abs(dv) > 1e-3)

    def test_general_interface(self):

        X_test = init_random_uniform(self.task.X_lower, self.task.X_upper, 1)

        a = self.acquisition_func(X_test, False)

        assert len(a.shape) == 2
        assert a.shape[0] == X_test.shape[0]
        assert a.shape[1] == 1

    def test_check_grads(self):
        x_ = np.array([[0.1]])
        assert check_grad(self.acquisition_func,
                lambda x: -self.acquisition_func(x, True)[1], x_) < 1e-3


if __name__ == "__main__":
    unittest.main()
