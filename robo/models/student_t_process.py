# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:47:47 2015

@author: aaron
"""

import sys
import StringIO
import numpy as np
import GPy

from robo.models.gpy_model import GPyModel


class StudentTProcess(GPyModel):
    """
    """
    def __init__(self, kernel, optimize=True, num_restarts=100, *args, **kwargs):
        self.kernel = kernel
        self.optimize = optimize
        self.num_restarts = num_restarts

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        assert X.size is not  0
        assert Y.size is not 0

        t_distribution = GPy.likelihoods.StudentT()
        laplace = GPy.inference.latent_function_inference.Laplace()

        self.m = GPy.core.GP(X, Y, kernel=self.kernel, inference_method=laplace, likelihood=t_distribution)
        self.m.constrain_positive('t_noise') 
        if self.optimize:
            self.m.optimize()