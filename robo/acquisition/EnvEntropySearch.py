'''
Created on Jun 8, 2015

@author: Aaron Klein
'''

import emcee
import numpy as np

from robo.acquisition.LogEI import LogEI
from robo.acquisition.EntropyMC import EntropyMC

from scipy import stats


class EnvEntropySearch(EntropyMC):
    '''
    classdocs
    '''

    def __init__(self, model, cost_model, X_lower, X_upper, compute_incumbent, is_env_variable, n_representer=10, n_hals_vals=100, n_func_samples=100, **kwargs):

        self.cost_model = cost_model
        self.n_dims = X_lower.shape[0]

        self.is_env_variable = is_env_variable

        super(EnvEntropySearch, self).__init__(model, X_lower, X_upper, compute_incumbent, Nb=n_representer, Nf=n_func_samples, Np=n_hals_vals)
        #TODO: Make sure LogEI gets only one model if we do MCMC sampling
        #self.proposal_measure = LogEI(self.model, X_lower, X_upper, compute_incumbent)
        #self._current_pmin()

#     def _current_pmin(self):
#         # Sample representer points
#         #TODO: I think we also want to add the incumbent to the representer points (as this is the most likely global minimum)?
#         self.representers = self._sample_representers(self.proposal_measure, self.n_representers, self.n_dims)
# 
#         # Compute log proposal values
#         self.log_proposal_vals = np.array([self.proposal_measure(self.representers[i]) for i in xrange(self.n_representers)])
# 
#         # Compute kullback leibler divergence between current pmin and the uniform distribution
#         self._current_pmin = self._compute_pmin(self.model, self.representers)
#         self.current_kl = self._loss_kl_div(self._current_pmin, self.log_proposal_vals)

    def update(self, model, cost_model):
        self.cost_model = cost_model
        super(EnvEntropySearch, self).update(model)

#         self.proposal_measure.update(model)
# 
#         self._current_pmin()

    def compute(self, X, derivative=False):

        # Compute kl divergence between the updated pmin and the uniform distribution
        #TODO: Fantasize the change of the model for X
#         pmin = self._compute_pmin(innovated_model, self.representers)
# 
#         # Predict the costs for this configuration
        cost = self.cost_model.predict(X)
#         # Compute acquisition value
#         kl = self._compute_kl_divergence(pmin, self.log_proposal_vals)

        new_pmin = self.change_pmin_by_innovation(X, self.f)
        H_old = np.sum(np.multiply(self.pmin, (self.logP + self.lmb)))
        H_new = np.sum(np.multiply(new_pmin, (np.log(new_pmin) + self.lmb)))

        loss = np.array([[-H_new + H_old]])
        acquisition_value = loss / cost

        return acquisition_value

    def update_representer_points(self):

        # Start from some random points
        #TODO: We might want to start from the incumbent here? Or maybe from a sobel grid?
#         p0 = np.array([np.random.rand(n_dim) for i in range(n_representers)])
# 
#         sampler = emcee.EnsembleSampler(n_representers, n_dim, proposal_measure)
# 
#         pos, prob, state = sampler.run_mcmc(p0, burnin_steps)
#         sampler.reset()
#         sampler.run_mcmc(pos, mcmc_steps)
#         representers = sampler.chain[:, -1, :]
        super(EnvEntropySearch, self).update_representer_points()

        # Project representer points to subspace
        self.zb[:, self.is_env_variable == 1] = self.X_upper[self.is_env_variable == 1]
 
#     def _compute_pmin(self, model, representers, num_func_samples=1000):
#         K_star = model.kernel.K(representers)
#         mean, _ = model.predict(representers, full_cov=False)
#         func_samples = np.random.multivariate_normal(mean, K_star, num_func_samples)
#         minimums = np.zeros(func_samples.shape)
#         idx = np.argmin(func_samples, axis=1)
#         minimums[np.arange(0, func_samples.shape[0], 1), idx] = 1
#         pmin = np.sum(minimums, axis=0) / float(representers.shape[0])
# 
#         return pmin

    def _loss_kl_div(self, pmin, log_proposal_vals):
        entropy_pmin = stats.entropy(pmin)
        entropy_log_proposal = stats.entropy(log_proposal_vals, pmin)
        loss = (entropy_pmin - entropy_log_proposal)
        return loss

