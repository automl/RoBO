'''
Created on Jun 8, 2015

@author: Aaron Klein
'''

import emcee
import numpy as np

from robo.acquisition.LogEI import LogEI
from robo.acquisition.Entropy import Entropy

from scipy import stats


class EnvEntropySearch(Entropy):
    '''
    classdocs
    '''

    def __init__(self, model, cost_model, X_lower, X_upper, compute_incumbent, is_env_variable, n_representer=10, n_hals_vals=100, n_func_samples=100, **kwargs):

        self.model = model
        self.cost_model = cost_model
        self.n_representers = n_representer
        self.n_func_samples = n_func_samples
        self.n_hals_vals = n_hals_vals
        self.X_lower = X_lower
        self.X_upper = X_upper
        self.n_dims = self.X_lower.shape[0]

        self.is_env_variable = is_env_variable

        #TODO: Make sure LogEI gets only one model if we do MCMC sampling
        self.proposal_measure = LogEI(self.model, X_lower, X_upper, compute_incumbent)
        self._current_pmin()

    def _current_pmin(self):
        # Sample representer points
        #TODO: I think we also want to add the incumbent to the representer points (as this is the most likely global minimum)?
        self.representers = self._sample_representers(self.proposal_measure, self.n_representers, self.n_dims)

        # Compute log proposal values
        self.log_proposal_vals = np.array([self.proposal_measure(self.representers[i]) for i in xrange(self.n_representers)])

        # Compute kullback leibler divergence between current pmin and the uniform distribution
        self._current_pmin = self._compute_pmin(self.model, self.representers)
        self.current_kl = self._loss_kl_div(self._current_pmin, self.log_proposal_vals)

    def update(self, model, cost_model):

        self.model = model
        self.cost_model = cost_model

        self.proposal_measure.update(model)

        self._current_pmin()

    def __call__(self, X, derivative=False):

        # Compute kl divergence between the updated pmin and the uniform distribution
        #TODO: Fantasize the change of the model for X
        pmin = self._compute_pmin(innovated_model, self.representers)

        # Predict the costs for this configuration
        cost = self.cost_model.predict(X)
        # Compute acquisition value
        kl = self._compute_kl_divergence(pmin, self.log_proposal_vals)

        acquisition_value = (self.current_kl - kl) / cost

        return acquisition_value

    def _sample_representers(self, proposal_measure, n_representers, n_dim, burnin_steps=100, mcmc_steps=100):

        # Start from some random points
        #TODO: We might want to start from the incumbent here? Or maybe from a sobel grid?
        p0 = np.array([np.random.rand(n_dim) for i in range(n_representers)])

        sampler = emcee.EnsembleSampler(n_representers, n_dim, proposal_measure)

        pos, prob, state = sampler.run_mcmc(p0, burnin_steps)
        sampler.reset()
        sampler.run_mcmc(pos, mcmc_steps)
        representers = sampler.chain[:, -1, :]

        # Project representer points to subspace
        representers[:, self.is_env_variable == 1] = self.X_upper[self.is_env_variable == 1]
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

    def _loss_kl_div(self, pmin, log_proposal_vals):
        entropy_pmin = stats.entropy(pmin)
        entropy_log_proposal = stats.entropy(log_proposal_vals, pmin)
        loss = (entropy_pmin - entropy_log_proposal)
        return loss

