'''
Created on Oct 14, 2015

@author: Aaron Klein
'''
import numpy as np
import scipy.stats as sps

from robo.priors.base_prior import BasePrior


class TophatPrior(BasePrior):
    def __init__(self, l_bound, u_bound):
        self.min = l_bound
        self.max = u_bound
        if not (self.max > self.min):
            raise Exception("Upper bound of Tophat prior must be greater than the lower bound!")

    def lnprob(self, theta):
        if np.any(theta < self.min) or np.any(theta > self.max):
            return -np.inf
        else:
            return 0

    def sample_from_prior(self, n_samples):
        return self.min + np.random.rand(n_samples) * (self.max - self.min)

    def gradient(self, theta):
        if np.any(theta < self.min) or np.any(theta > self.max):
            return -np.inf
        else:
            return 0

# Copied from Spearmint
class HorseshoePrior(BasePrior):
    def __init__(self, scale=0.1):
        self.scale = scale

    # THIS IS INEXACT
    def lnprob(self, theta):
        if np.any(theta == 0.0):
            return np.inf  # POSITIVE infinity (this is the "spike")
        # We don't actually have an analytical form for this
        # But we have a bound between 2 and 4, so I just use 3.....
        # (or am I wrong and for the univariate case we have it analytically?)
        return np.log(np.log(1 + 3.0 * (self.scale / np.exp(theta)) ** 2))

    def sample_from_prior(self, n_samples):

        lamda = np.abs(np.random.standard_cauchy(size=n_samples))

        return np.log(np.abs(np.random.randn() * lamda * self.scale))


class LognormalPrior(BasePrior):
    def __init__(self, sigma, mean=0):
        self.sigma = sigma
        self.mean = mean

    def lnprob(self, theta):
        return sps.lognorm.logpdf(theta, self.sigma, loc=self.mean)

    def sample_from_prior(self, n_samples):
        return np.random.lognormal(mean=self.mean,
                                   sigma=self.sigma,
                                   size=n_samples)
