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

logger = logging.getLogger(__name__)


class GaussianProcess(BaseModel):
    
    def __init__(self, kernel, mean=0, *args, **kwargs):
        self.kernel = kernel
        self.model = None
        self.mean = mean
        
    def scale(self, x, new_min, new_max, min, max):
        return ((new_max - new_min) * (x -min) / (max - min)) + new_min


    def train(self, X, Y, do_optimize=True):
        self.X = X
        #self.Y = self.scale(Y, 0, 100, np.min(Y, axis=0), np.max(Y, axis=0))
        self.Y = Y
        
        # Use the mean of the data as mean for the GP
        self.mean = np.mean(Y, axis=0)
        self.model = george.GP(self.kernel, mean=self.mean)
        
        # Precompute the covariance
        yerr = 1e-25
        while(True):
            try:
                self.model.compute(self.X, yerr=yerr)
                break
            except np.linalg.LinAlgError:
                yerr *= 10
                logger.error("Cholesky decomposition for the covariance matrix of the GP failed. Add %s noise on the diagonal." % yerr)
        
        
        if do_optimize:
            self.hypers = self.optimize()
            logger.debug("HYPERS: " + str(self.hypers))
            self.model.kernel[:] = self.hypers
        else:
            self.hypers = self.model.kernel[:]

    def get_noise(self):
        # Assumes a kernel of the form amp * (kernel1 + noise_kernel)
        # FIXME: How to determine the noise of george gp?
        assert self.kernel.k2.k2.kernel_type == 1
        return self.kernel.k2.k2.pars[0]
    
    def nll(self, p):
        # Specify bounds to keep things sane
        if np.any((-10 > p) + (p > 10)):
            return 1e25
    
        self.model.kernel[:] = p
        ll = self.model.lnlikelihood(self.Y[:, 0], quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(self, p):
        self.model.kernel[:] = p
        return -self.model.grad_lnlikelihood(self.Y[:, 0], quiet=True)
        
    def optimize(self):
        # Start optimization  from the previous hyperparameter configuration
        p0 = self.model.kernel.vector
        results = optimize.minimize(self.nll, p0, jac=self.grad_nll)
        
        return results.x
    

    def predict_variance(self, X1, X2):
        # Predict the variance between two test points X1, X2 by Sigma(X1, X2) = k_X1,X2 - k_X1,X * (K_X,X + simga^2*I)^-1 * k_X,X2)
        var = self.kernel.value(X1, X2) - np.dot(self.kernel.value(X1, self.X), self.model.solver.apply_inverse(self.kernel.value(self.X, X2)))
        return var

    def predict(self, X, **kwargs):
        if self.model is None:
            logger.error("The model has to be trained first!")
            raise ValueError

        mu, var = self.model.predict(self.Y[:, 0], X)

        return mu, var
