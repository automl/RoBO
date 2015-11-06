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
                 burnin_steps=2000, scaling=False, *args, **kwargs):
        """
        GaussianProcess model based on the george GP library that uses MCMC
        sampling to marginalise the hyperparmeter. If you use this class
        make sure that to use the IntegratedAcqusition function to integrate
        over the GP's hyperparameter as proposed by Snoek et al.

        Parameters
        ----------
        kernel : george kernel object

        prior : prior object

        n_hypers : int

        chain_length : int

        burnin_steps : int

        scaling : boolean
        """

        self.kernel = kernel
        if prior is None:
            prior = lambda x : 0
        else:
            self.prior = prior
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        
        # This flag is only need for environmental entropy search to transform s into (1 - s) ** 2
        self.scaling = scaling
        
                
    def scale(self, x, new_min, new_max, min, max):
        return ((new_max - new_min) * (x -min) / (max - min)) + new_min
        
    def train(self, X, Y, do_optimize=True, **kwargs):        
        """
        
        

        Parameters
        ----------
        X : 

        Y : 

        do_optimize : int
        """

        
        self.X = X
        self.Y = Y
        
        
        # Transform s to (1 - s) ** 2 only necessary for environmental entropy search
        if self.scaling:
            self.X = np.copy(X)
            self.X[:, -1] = (1 - self.X[:, -1]) ** 2

        # Use the mean of the data as mean for the GP
        mean = np.mean(Y, axis=0)
        self.gp = george.GP(self.kernel, mean=mean)
        
        # Precompute the covariance
        yerr = 1e-25
        while(True):
            try:
                self.gp.compute(self.X, yerr=yerr)
                break
            except np.linalg.LinAlgError:
                yerr *= 10
                logging.error("Cholesky decomposition for the covariance matrix of the GP failed. Add %s noise on the diagonal." % yerr)

        if do_optimize:
            # We have one walker for each hyperparameter configuration
            self.sampler = emcee.EnsembleSampler(self.n_hypers, len(self.kernel.pars), self.loglikelihood)
            
            # Do a burn-in in the first iteration
            if not self.burned:
                # Initialize the walkers by sampling from the prior 
                self.p0 = self.prior.sample_from_prior(self.n_hypers)
                # Run mcmc sampling
                self.p0, _, _ = self.sampler.run_mcmc(self.p0, self.burnin_steps)
                
                self.burned = True
    
            # Start sampling
            pos, prob, state = self.sampler.run_mcmc(self.p0, self.chain_length)
            
            # Save the current position, it will be the startpoint in the next iteration
            self.p0 = pos
            
            # Take the last samples from each walker
            self.hypers = self.sampler.chain[:, -1]
            
            self.models = []
            logging.info("Hypers: %s" % self.hypers)
            for sample in self.hypers:
                
                # Instantiate a model for each hyperparam configuration
                #TODO: Just keep one model and replace the hypers every time we need them
                kernel = deepcopy(self.kernel)
                kernel.pars = np.exp(sample)

                model = GaussianProcess(kernel)
                model.train(self.X, self.Y, do_optimize=False)
                self.models.append(model)
        else:
            self.hypers = self.gp.kernel[:]
                   
    def loglikelihood(self, theta):
        """

        Parameters
        ----------
        theta : 

        """

        # Bound the hyperparameter space to keep things sane. Note all hyperparameters live on a log scale
        if np.any((-40 > theta) + (theta > 40)):
            return -np.inf
        
        # Update the kernel and compute the lnlikelihood. Hyperparameters are all on a log scale
        self.gp.kernel.pars = np.exp(theta[:])
        
        return self.prior.lnprob(theta) + self.gp.lnlikelihood(self.Y[:, 0], quiet=True)

    def predict(self, X, **kwargs):
        """
        Returns the predictive mean and variance at X

        Parameters
        ----------
        X : 

        """

        
        if self.scaling:
            X[:, -1] = (1 - X[:, -1]) ** 2
        mu = np.zeros([self.n_hypers])
        var = np.zeros([self.n_hypers])
        for i, model in enumerate(self.models):
            mu[i], var[i] = model.predict(X)
        
        return np.array([mu.mean()]), np.array([[var.mean()]])
