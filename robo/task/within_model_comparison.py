# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 13:13:06 2015

@author: Aaron Klein
"""

import GPy
import numpy as np
import scipy.linalg as sla
from robo.task.base_task import BaseTask


class WithinModelComparison(BaseTask):

    def __init__(self, seed=42):
        X_lower = np.array([0, 0])
        X_upper = np.array([1, 1])
        rng = np.random.RandomState(seed)
        kern = GPy.kern.RBF(2, lengthscale=0.1, variance=1.0)
        xstar = rng.rand(100, 2)
        K = kern.K(xstar, xstar)
        L = sla.cholesky(K)
        sigma = rng.randn(100)
        f = np.dot(L, sigma)
        self.gp = GPy.models.GPRegression(xstar, f[:, np.newaxis], kern)
        best = np.argmin(f)
        fopt = f[best]
        opt = xstar[best]
        super(WithinModelComparison, self).__init__(X_lower, X_upper, opt, fopt)

    def objective_function(self, x):
        mu, _ = self.gp.predict(x)

        return mu

    def evaluate_test(self, x):
        return self.objective_function(x)
