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

    def __init__(self, kernel, prior=None, mean=0,
                 yerr=1e-25, *args, **kwargs):

        self.kernel = kernel
        self.model = None
        self.mean = mean
        self.prior = prior
        self.yerr = yerr

    def scale(self, x, new_min, new_max, old_min, old_max):
        return ((new_max - new_min) *
                (x - old_min) / (old_max - old_min)) + new_min

    def train(self, X, Y, do_optimize=True):
        self.X = X
        self.Y = Y

        # Use the mean of the data as mean for the GP
        self.mean = np.mean(Y, axis=0)
        self.model = george.GP(self.kernel, mean=self.mean)

        # Precompute the covariance
        while(True):
            try:
                self.model.compute(self.X, yerr=self.yerr)
                break
            except np.linalg.LinAlgError:
                self.yerr *= 10
                logger.error(
                    "Cholesky decomposition for the covariance matrix \
                    of the GP failed. \
                    Add %s noise on the diagonal." %
                    self.yerr)

        if do_optimize:
            self.hypers = self.optimize()
            logger.info("HYPERS: " + str(self.hypers))
            self.model.kernel[:] = self.hypers
        else:
            self.hypers = self.model.kernel[:]

    def get_noise(self):
        # Assumes a kernel of the form amp * (kernel1 + noise_kernel)
        # FIXME: How to determine the noise of george gp in general?
        assert self.kernel.k2.k2.kernel_type == 1
        return self.kernel.k2.k2.pars[0]

    def nll(self, theta):
        # Specify bounds to keep things sane
        if np.any((-10 > theta) + (theta > 10)):
            return 1e25

        self.model.kernel[:] = theta
        ll = self.model.lnlikelihood(self.Y[:, 0], quiet=True)

        # Add prior
        ll += self.prior.lnprob(theta)

        # We add a minus here because scipy is minimizing
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(self, theta):
        self.model.kernel[:] = theta

        gll = self.model.grad_lnlikelihood(self.Y[:, 0], quiet=True)
        gll += self.prior.gradient(theta)
        return -gll

    def optimize(self):
        # Start optimization  from the previous hyperparameter configuration
        p0 = self.model.kernel.vector
        results = optimize.minimize(self.nll, p0, jac=self.grad_nll)

        return results.x

    def predict_variance(self, X1, X2):
        # Predict the variance between two test points X1, X2 by
        # Sigma(X1, X2) = k_X1,X2 - k_X1,X * (K_X,X + sigma^2*I)^-1 * k_X,X2)
        x_ = np.concatenate((X1, X2))
        _, var = self.predict(x_)
        var = var[:-1, -1, np.newaxis]

        return var

    def predict(self, X, **kwargs):
        if self.model is None:
            logger.error("The model has to be trained first!")
            raise ValueError

        mu, var = self.model.predict(self.Y[:, 0], X)

        return mu, var

    def sample_functions(self, X_test, n_funcs=1):
        return self.model.sample_conditional(self.Y[:, 0], X_test, n_funcs)
