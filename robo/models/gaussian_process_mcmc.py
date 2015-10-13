'''
Created on Oct 12, 2015

@author: Aaron Klein
'''

import logging
import george
import emcee
import numpy as np
from scipy import optimize
from copy import deepcopy

from robo.models.base_model import BaseModel
from robo.models.gaussian_process import GaussianProcess


class GaussianProcessMCMC(BaseModel):
    
    def __init__(self, kernel, lnprior=None, n_hypers=20, chain_length=2000, burnin_steps=2000, *args, **kwargs):
        self.kernel = kernel
        if lnprior is None:
            lnprior = lambda x : 0
        else:
            self.lnprior = lnprior
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        
    def train(self, X, Y, do_optimize=True):
        self.X = X
        self.Y = Y
        
        # Use the mean of the data as mean for the GP
        mean = np.mean(Y, axis=0)
        self.gp = george.GP(self.kernel, mean=mean)
        
        # Precompute the covariance
        self.gp.compute(self.X)

        if do_optimize:
            # Initialize the walkers. We have one walker for each hyperparameter configuration
            self.sampler = emcee.EnsembleSampler(self.n_hypers, len(self.kernel), self.lnprob)
            p0 = [np.log(self.kernel.pars) + 1e-4 * np.random.randn(len(self.kernel)) for i in range(self.n_hypers)]
    
            # Do a burn-in in the first iteration
            if not self.burned:
                self.p0, _, _ = self.sampler.run_mcmc(p0, self.burnin_steps)
                self.burned = True
    
            # Start sampling
            pos, prob, state = self.sampler.run_mcmc(self.p0, self.chain_length)
            
            # Save the current position, it will be the startpoint in the next iteration
            self.p0 = pos
            
            # Take the last samples from each walker
            self.hypers = self.sampler.chain[:, -1]
            
            self.models = []
            for sample in self.hypers:
                # Instantiate a model for each hyperparam configuration
                #TODO: Just keep one model and replace the hypers every time we need them
                kernel = deepcopy(self.kernel)
                
                model = GaussianProcess(kernel)
                model.train(self.X, self.Y, do_optimize=False)
                self.models.append(model)
        else:
            self.hypers = self.gp.kernel[:]
                   
    def lnprob(self, p):
        # Update the kernel and compute the lnlikelihood.
        self.gp.kernel.pars = np.exp(p)
        return self.lnprior(p) + self.gp.lnlikelihood(self.Y[:, 0], quiet=True)

    def predict(self, X):
        mu = np.zeros([self.n_hypers])
        var = np.zeros([self.n_hypers])
        for i, model in enumerate(self.models):
            mu[i], _ = model.predict(X)
        return np.array([mu.mean()]), np.array([[mu.var()]])
