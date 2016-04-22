# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 13:13:06 2015

@author: Aaron Klein
"""

import george
import numpy as np
import scipy.linalg as sla

from robo.models.gaussian_process import GaussianProcess
from robo.task.base_task import BaseTask


class WithinModelComparison(BaseTask):

    def __init__(self, seed=42):
        X_lower = np.array([0, 0])
        X_upper = np.array([1, 1])
        rng = np.random.RandomState(seed)

        cov_amp = 1.0
        mat_kernel = george.kernels.Matern52Kernel(np.ones([2]) * 0.1,
                                               ndim=2)

        kernel = cov_amp * mat_kernel

        self.xstar = rng.rand(1000, 2)
        K = kernel.value(self.xstar)
        L = sla.cholesky(K)
        sigma = rng.randn(1000)
        self.f = np.dot(L, sigma)
        self.gp = GaussianProcess(kernel, yerr=0.0)
        self.gp.train(self.xstar, self.f[:, np.newaxis], do_optimize=False)
        self.gp.train(self.xstar, self.f[:, np.newaxis], do_optimize=False)
        best = np.argmin(self.f)
        fopt = self.f[best]
        opt = self.xstar[best]
        super(WithinModelComparison, self).__init__(X_lower, X_upper, opt, fopt)

    def objective_function(self, x):
        noise = 1e-3 * np.random.randn()
        mu, _ = self.gp.predict(x)

        return mu + noise

    def evaluate_test(self, x):
        return self.objective_function(x)
