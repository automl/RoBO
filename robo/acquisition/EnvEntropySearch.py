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

    def update(self, model, cost_model):
        self.cost_model = cost_model
        super(EnvEntropySearch, self).update(model)

    def __call__(self, X, derivative=False):

        # Predict the costs for this configuration
        cost = self.cost_model.predict(X)[0]

        # Compute fantasized pmin
        new_pmin = self.change_pmin_by_innovation(X, self.f)

        # Compute acquisition value
        H_old = np.sum(np.multiply(self.pmin, (self.logP + self.lmb)))
        H_new = np.sum(np.multiply(new_pmin, (np.log(new_pmin) + self.lmb)))

        loss = np.array([[-H_new + H_old]])

        acquisition_value = loss / cost

        return acquisition_value

    def update_representer_points(self):

        #TODO: We might want to start the sampling of the representer points from the incumbent here? Or maybe from a sobel grid?
        super(EnvEntropySearch, self).update_representer_points()

        # Project representer points to subspace
        self.zb[:, self.is_env_variable == 1] = self.X_upper[self.is_env_variable == 1]
