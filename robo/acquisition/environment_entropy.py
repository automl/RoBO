'''
Created on Jun 8, 2015

@author: Aaron Klein
'''
import numpy as np

from robo.acquisition.entropy_mc import EntropyMC
from robo.acquisition.entropy import Entropy
from robo.recommendation.optimize_posterior import env_optimize_posterior_mean_and_std
from _functools import partial


class EnvironmentEntropy(Entropy):

    def __init__(self, model, cost_model, X_lower, X_upper,
                 compute_incumbent,
                 is_env_variable,
                 n_representer=50, **kwargs):
        """
        Adaption of the original Entropy Search by Hennig et al for
        environmental variable.

        Parameters
        ----------
        model : Model object
            Models the objective function. The model has to be a
            Gaussian process. If MCMC sampling of the model's hyperparameter is
            performed, make sure that the acquistion_func is of an instance of
            IntegratedAcquisition to marginalise over the GP's hyperparameter.
        cost_model : model
            Models the cost function. The model has to be a Gaussian Process.
        X_lower : (D) numpy array
            Specified the lower bound of the input space. Each entry
            corresponds to one dimension.
        X_upper : (D) numpy array
            Specified the upper bound of the input space. Each entry
            corresponds to one dimension.
        compute_incumbent : function
            Computes the current incumbent.
        is_env_variable : (D) numpy array
            Specifies which input dimension is an environmental variable. If
            the i-th input is an environmental variable than the i-th entry has
            to be 1 and 0 otherwise.
        n_representers : int, optional
            The number of representer points to discretize the input space and
            to compute pmin.
        """
        self.cost_model = cost_model
        self.n_dims = X_lower.shape[0]

        self.is_env_variable = is_env_variable

        if compute_incumbent is env_optimize_posterior_mean_and_std:
            compute_incumbent = partial(compute_incumbent,
                                        is_env=is_env_variable)

        super(EnvironmentEntropy, self).__init__(model, X_lower, X_upper,
                                                 compute_inc=compute_incumbent,
                                                 Nb=n_representer)

    def update(self, model, cost_model):
        self.cost_model = cost_model
        super(EnvironmentEntropy, self).update(model)

    def compute(self, X, derivative=False):
        """
        Computes the acquisiton value for a single point.

        Parameters
        ----------
        X : (1, D) numpy array
            The input point for which the acquisition functions is computed.
        derivative : bool, optional
            If it is equal to True also the derivatives with respect to X is
            computed.

        Returns
        -------
        acquisition_value: numpy array
            The acquisition value computed for X.
        grad : numpy array
            The computed gradient of the acquisition function at X. Only
            returned if derivative==True
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        # Predict the log costs for this configuration
        log_cost = self.cost_model.predict(X)[0]

        if derivative:
            #dh, g = super(EnvironmentEntropy, self).compute(X,
            #                                    derivative=derivative)

            #dmu = self.cost_model.predictive_gradients(X[:, self.is_env_variable == 1])[0]
            #log_cost = (log_cost + 1e-8)
            #acquisition_value = dh / log_cost
            #grad = g * log_cost + dmu * dh

            #return acquisition_value, grad
            raise("Not implemented")
        else:
            dh = super(EnvironmentEntropy, self).compute(X,
                                                derivative=derivative)
            acquisition_value = dh / np.exp(log_cost)

            return acquisition_value

    def update_representer_points(self):

        super(EnvironmentEntropy, self).update_representer_points()

        # Project representer points to subspace
        self.zb[:, self.is_env_variable == 1] = self.X_upper[self.is_env_variable == 1]


# class EnvEntropySearchMC(EntropyMC):
#     '''
#         Environment Entropy Search
#     '''
#
#     def __init__(self, model, cost_model, X_lower, X_upper, compute_incumbent, is_env_variable, n_representer=10, n_hals_vals=100, n_func_samples=100, **kwargs):
#
#         self.cost_model = cost_model
#         self.n_dims = X_lower.shape[0]
#
#         self.is_env_variable = is_env_variable
#
#         if compute_incumbent is env_optimize_posterior_mean_and_std:
#             compute_incumbent = partial(compute_incumbent, is_env=is_env_variable)
#
#         super(EnvEntropySearchMC, self).__init__(model, X_lower, X_upper, compute_incumbent, Nb=n_representer, Nf=n_func_samples, Np=n_hals_vals)
#
#     def update(self, model, cost_model):
#         self.cost_model = cost_model
#         super(EnvEntropySearchMC, self).update(model)
#
#     def __call__(self, X, derivative=False):
#
#        # Predict the costs for this configuration
#         cost = self.cost_model.predict(X)[0]
#
#         # Compute fantasized pmin
#         new_pmin = self.change_pmin_by_innovation(X, self.f)
#
#         # Compute acquisition value
#         H_old = np.sum(np.multiply(self.pmin, (self.logP + self.lmb)))
#         H_new = np.sum(np.multiply(new_pmin, (np.log(new_pmin) + self.lmb)))
#
#         loss = np.array([[-H_new + H_old]])
#
#         acquisition_value = loss / (cost + 1e-8)
#
#         return acquisition_value
#
#    def update_representer_points(self):
#
#         #TODO: We might want to start the sampling of the representer points from the incumbent here? Or maybe from a sobel grid?
#         #TODO: Sample only in the subspace
#         super(EnvEntropySearchMC, self).update_representer_points()
#
#         # Project representer points to subspace
#         self.zb[:, self.is_env_variable == 1] = self.X_upper[self.is_env_variable == 1]
