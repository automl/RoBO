'''
Created on Oct 14, 2015

@author: Aaron Klein
'''
import numpy as np

from robo.priors.base_prior import BasePrior, TophatPrior, \
    LognormalPrior, HorseshoePrior


class DefaultPrior(BasePrior):

    def __init__(self, n_dims):

        # The number of hyperparameters
        self.n_dims = n_dims

        # Prior for the Matern52 lengthscales
        self.tophat = TophatPrior(-2, 2)

        # Prior for the covariance amplitude
        self.ln_prior = LognormalPrior(mean=0.0, sigma=1.0)

        # Prior for the noise
        self.horseshoe = HorseshoePrior(scale=0.1)

    def lnprob(self, theta):
        lp = 0
        # Covariance amplitude
        lp += self.ln_prior.lnprob(theta[0])
        # Lengthscales
        lp += self.tophat.lnprob(theta[1:-1])
        # Noise
        lp += self.horseshoe.lnprob(theta[-1])

        return lp

    def sample_from_prior(self, n_samples):
        p0 = np.zeros([n_samples, self.n_dims])
        # Covariance amplitude
        p0[:, 0] = self.ln_prior.sample_from_prior(n_samples)[:, 0]
        # Lengthscales
        ls_sample = np.array([self.tophat.sample_from_prior(n_samples)[:, 0]
                              for _ in range(1, (self.n_dims - 1))]).T
        p0[:, 1:(self.n_dims - 1)] = ls_sample
        # Noise
        p0[:, -1] = self.horseshoe.sample_from_prior(n_samples)[:, 0]
        return p0

    def gradient(self, theta):
        #TODO: Implement real gradient here
        return np.zeros([theta.shape[0]])