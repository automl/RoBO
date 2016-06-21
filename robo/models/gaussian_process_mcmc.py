'''
Created on Oct 12, 2015

@author: Aaron Klein
'''

import logging
import george
import emcee
import numpy as np
from copy import deepcopy

from robo.models.base_model import BaseModel
from robo.models.gaussian_process import GaussianProcess

logger = logging.getLogger(__name__)


class GaussianProcessMCMC(BaseModel):

    def __init__(self, kernel, prior=None, n_hypers=20, chain_length=2000,
                 burnin_steps=2000, basis_func=None, dim=None,
                 normalize_output=False, *args, **kwargs):
        """
        GaussianProcess model based on the george GP library that uses MCMC
        sampling to marginalise over the hyperparmeters. If you use this class
        make sure that you also use the IntegratedAcqusition function to
        integrate over the GP's hyperparameter as proposed by Snoek et al.

        Parameters
        ----------
        kernel : george kernel object
            Specifies the kernel that is used for all Gaussian Process
        prior : prior object
            Defines a prior for the hyperparameters of the GP. Make sure that
            it implements the Prior interface. During MCMC sampling the
            lnlikelihood is multiplied with the prior.
        n_hypers : int
            The number of hyperparameter samples. This also determines the
            number of walker for MCMC sampling as each walker will
            return one hyperparameter sample.
        chain_length : int
            The length of the MCMC chain. We start n_hypers walker for
            chain_length steps and we use the last sample
            in the chain as a hyperparameter sample.
        burnin_steps : int
            The number of burnin steps before the actual MCMC sampling starts.
        """

        self.kernel = kernel
        if prior is None:
            self.prior = lambda x: 0
        else:
            self.prior = prior
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        self.basis_func = basis_func
        self.dim = dim
        self.models = []
        self.normalize_output = normalize_output        

    def _scale(self, x, new_min, new_max, old_min, old_max):
        return ((new_max - new_min) * (x - old_min) / (old_max - old_min)) + new_min

    def train(self, X, Y, do_optimize=True, **kwargs):
        """
        Performs MCMC sampling to sample hyperparameter configurations from the
        likelihood and trains for each sample a GP on X and Y

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        Y: np.ndarray (N, 1)
            The corresponding target values.
        do_optimize: boolean
            If set to true we perform MCMC sampling otherwise we just use the
            hyperparameter specified in the kernel.
        """
        self.X = X

        # For Fabolas we transform s to (1 - s)^2

        if self.basis_func is not None:
            self.X = deepcopy(X)
            self.X[:, self.dim] = self.basis_func(self.X[:, self.dim])
        
        self.Y = Y
        if self.normalize_output:
            self.Y_mean = np.mean(Y)
            self.Y_std = np.std(Y)
            self.Y = (Y - self.Y_mean) / self.Y_std
        

        # Use the mean of the data as mean for the GP
        mean = np.mean(self.Y, axis=0)
        self.gp = george.GP(self.kernel, mean=mean)


        if do_optimize:
            # We have one walker for each hyperparameter configuration
            self.sampler = emcee.EnsembleSampler(self.n_hypers,
                                                 len(self.kernel.pars) + 1,
                                                 self.loglikelihood)

            # Do a burn-in in the first iteration
            if not self.burned:
                # Initialize the walkers by sampling from the prior
                self.p0 = self.prior.sample_from_prior(self.n_hypers)
                # Run MCMC sampling
                self.p0, _, _ = self.sampler.run_mcmc(self.p0,
                                                      self.burnin_steps)

                self.burned = True

            # Start sampling
            pos, _, _ = self.sampler.run_mcmc(self.p0,
                                              self.chain_length)

            # Save the current position, it will be the startpoint in
            # the next iteration
            self.p0 = pos

            # Take the last samples from each walker
            self.hypers = self.sampler.chain[:, -1]

            self.models = []
        else:
            self.hypers = [self.gp.kernel[:]]


        for sample in self.hypers:

            # Instantiate a model for each hyperparameter configuration
            kernel = deepcopy(self.kernel)
            kernel.pars = np.exp(sample[:-1])
            noise = np.exp(sample[-1])
            model = GaussianProcess(kernel,
                                    basis_func=self.basis_func,
                                    dim=self.dim,
                                    normalize_output=self.normalize_output,
                                    noise=noise)
            model.train(X, Y, do_optimize=False)
            self.models.append(model)

    def loglikelihood(self, theta):
        """
        Return the loglikelihood (+ the prior) for a hyperparameter
        configuration theta.

        Parameters
        ----------
        theta : np.ndarray(H)
            Hyperparameter vector. Note that all hyperparameter are
            on a log scale.

        Returns
        ----------
        float
            lnlikelihood + prior
        """

        # Bound the hyperparameter space to keep things sane. Note all
        # hyperparameters live on a log scale
        if np.any((-20 > theta) + (theta > 20)):
            return -np.inf
            
        # The last entry is always the noise
        sigma_2 = np.exp(theta[-1])
        # Update the kernel and compute the lnlikelihood.
        self.gp.kernel.pars = np.exp(theta[:-1])
        
        try:
            self.gp.compute(self.X, yerr=np.sqrt(sigma_2))
        except:
            return -np.inf

        return self.prior.lnprob(theta) + self.gp.lnlikelihood(self.Y[:, 0],
                                                                quiet=True)

    def predict(self, X, **kwargs):
        r"""
        Returns the predictive mean and variance of the objective function
        at X average over all hyperparameter samples.
        The mean is computed by:
        :math \mu(x) = \frac{1}{M}\sum_{i=1}^{M}\mu_m(x)
        And the variance by:
        :math \sigma^2(x) = (\frac{1}{M}\sum_{i=1}^{M}(\sigma^2_m(x) + \mu_m(x)^2) - \mu^2

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input test points

        Returns
        ----------
        np.array(N,1)
            predictive mean
        np.array(N,1)
            predictive variance

        """

        X_test = X
        mu = np.zeros([len(self.models), X_test.shape[0]])
        var = np.zeros([len(self.models), X_test.shape[0]])
        for i, model in enumerate(self.models):
            mu[i], var[i] = model.predict(X_test)

        # See the algorithm runtime prediction paper by Hutter et al
        # for the derivation of the total variance
        m = np.array([[mu.mean()]])
        v = np.mean(mu ** 2 + var) - m ** 2

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v[np.diag_indices(v.shape[0])] = \
                    np.clip(v[np.diag_indices(v.shape[0])],
                            np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        return m, v
