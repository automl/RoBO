'''
Created on Oct 12, 2015

@author: Aaron Klein
'''

import logging
import george
import emcee
import numpy as np
from scipy import optimize


from robo.models.base_model import BaseModel


class GaussianProcess(BaseModel):
    
    def __init__(self, kernel, *args, **kwargs):
        self.kernel = kernel
        self.model = None
        
    def train(self, X, Y, do_optimize=True):
        self.X = X
        self.Y = Y
        
        # Use the mean of the data as mean for the GP
        mean = np.mean(Y, axis=0)
        self.model = george.GP(self.kernel, mean=mean)
        
        # Precompute the covariance
        self.model.compute(self.X)
        
        if do_optimize:
            self.hypers = self.optimize()
            logging.info("HYPERS: " + str(self.hypers))
            self.model.kernel[:] = self.hypers
        else:
            self.hypers = self.model.kernel[:]

    def get_noise(self):
        # Assumes a kernel of the form amp * (kernel1 + noise_kernel)
        # FIXME: How to determine the noise of george gp?
        return self.kernel.k2.k2.pars[0]
    
    def nll(self, p):
        # Specify bounds to keep things sane
        if np.any((-10 > x) + (x > 10)):
            return 1e25
    
        self.model.kernel[:] = p
        ll = self.model.lnlikelihood(self.Y[:, 0], quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(self, p):
        self.model.kernel[:] = p
        return -self.model.grad_lnlikelihood(self.Y[:, 0], quiet=True)
        
    def optimize(self):
        #TODO: Maybe we should start from a random point here, as this might be more likely to get stuck in a local minima
        p0 = self.model.kernel.vector
        results = optimize.minimize(self.nll, p0, jac=self.grad_nll)
        
        return results.x
    

    def predict_variance(self, X1, X2):
        # Predict the variance between two test points X1, X2 by Sigma(X1, X2) = k_X1,X2 - k_X1,X * (K_X,X + simga^2*I)^-1 * k_X,X2)
        var = self.kernel.value(X1, X2) - np.dot(self.kernel.value(X1, self.X), self.model.solver.apply_inverse(self.kernel.value(self.X, X2)))
        return var

    def predict(self, X, **kwargs):
        if self.model is None:
            logging.error("The model has to be trained first!")
            raise ValueError
        mu, var = self.model.predict(self.Y[:, 0], X)
        return mu, var
