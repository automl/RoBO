'''
Created on Jun 8, 2015

@author: Aaron Klein
'''

import emcee
import numpy as np

from robo.acquisition.LogEI import LogEI
from robo.acquisition.Entropy import Entropy


class EnvEntropySearch(Entropy):
    '''
    classdocs
    '''

    def __init__(self, model, cost_model, n_representer, n_hals_vals, n_func_samples, X_lower, X_upper, compute_incumbent, **kwargs):

        self.model = model
        self.cost_model = cost_model
        self.n_representers = n_representer
        self.n_func_samples = n_func_samples
        self.n_hals_vals = n_hals_vals

        self.proposal_measure = LogEI(self.model, X_lower, X_upper, compute_incumbent)

    def update(self, model, cost_model):

        self.model = model
        self.cost_model = cost_model

        self.proposal_measure.update(model)

        # Sample representer points
        #TODO: project representer points
        self.representers = self._sample_representers(self.proposal_measure, self.n_representers, n_dim)

        # Compute log propsal vals
        self.log_proposal_vals = np.array([self.proposal_measure(self.representers[i]) for i in xrange(self.n_representers)])

        # Compute kl divergence between current pmin and the uniform distribution
        pmin = self._compute_pmin(self.model, self.representers)
        self.current_k = self._compute_kl_divergence(pmin, self.log_proposal_vals)

    def __call__(self, X, derivative=False):

        # Compute kl divergence between the updated pmin and the uniform distribution
        pmin = self._compute_pmin(self.model, self.representers)

        # Predict the costs for this configuration
        cost = self.cost_model.predict(X)
        # Compute acquisition value
        kl = self._compute_kl_divergence(pmin, self.log_proposal_vals)

        acquisition_value = (self.current_kl - kl) / cost

        return acquisition_value

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
