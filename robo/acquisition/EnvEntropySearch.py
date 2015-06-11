'''
Created on Jun 8, 2015

@author: Aaron Klein
'''

import emcee
import numpy as np

from Entropy import Entropy


class EnvEntropySearch(Entropy):
    '''
    classdocs
    '''

    def __init__(self, model, cost_model, n_representer, n_hals_vals, n_func_samples, **kwargs):
        '''
        Constructor
        '''
        self.model = model
        self.cost_model = cost_model

        # Sample representer points

        # Compute kl divergence between current pmin and the uniform distribution


    def update(self, model):
        """
        """
        self.model = model

    def __call__(self, X, derivative=False):
        """
        """
        # Compute kl divergence between the updated pmin and the uniform distribution

        # Predict the costs for this configuration

        # Compute acquisition value

        raise NotImplementedError()

    def _sample_representers(self, proposal_measure, n_representers, n_dim, burnin_steps=100, mcmc_steps=100):

        p0 = [np.random.rand(n_dim) for i in range(n_representers)]

        sampler = emcee.EnsembleSampler(n_representers, n_dim, proposal_measure)

        pos, prob, state = sampler.run_mcmc(p0, burnin_steps)
        sampler.reset()
        sampler.run_mcmc(pos, mcmc_steps)
        representers = sampler.chain[:, -1, :]
        return representers

    def _compute_pmin(self, model, representers, num_func_samples=1000):
        K_star = model.kernel.K(representers)
        mean, _ = model.predict(representers, full_cov=False)
        func_samples = np.random.multivariate_normal(mean, K_star, num_func_samples)
        minimums = np.zeros(func_samples.shape)
        idx = np.argmin(func_samples, axis=1)
        minimums[np.arange(0, func_samples.shape[0], 1), idx] = 1
        pmin = np.sum(minimums, axis=0) / float(representers.shape[0])

        return pmin

    def _compute_kl_divergence(self, pmin, log_proposal_vals):
        entropy_pmin = -np.dot(pmin, np.log(pmin + 1e-50))
        log_proposal = np.dot(log_proposal_vals, pmin)
        kl_divergence = (entropy_pmin - log_proposal)
        return kl_divergence
