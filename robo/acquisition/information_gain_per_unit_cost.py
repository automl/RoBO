import emcee
import numpy as np

from robo.acquisition.information_gain import InformationGain


class InformationGainPerUnitCost(InformationGain):

    def __init__(self, model, cost_model,
                 X_lower, X_upper,
                 is_env_variable,
                 n_representer=50, **kwargs):
        """
        Information gain per unit cost as described in Swersky et al. [1] which
        computes the information gain of a configuration divided by it's cost.
        
        This implementation slightly differs from the implementation of
        Swersky et al. as it additionally adds the optimization overhead to
        the cost. You can simply set the optimization overhead to 0 to obtain
        the original formulation.
        
        [1] Swersky, K., Snoek, J., and Adams, R.
            Multi-task Bayesian optimization.
            In Proc. of NIPS 13, 2013.

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

        self.is_env = is_env_variable

        super(InformationGainPerUnitCost, self).__init__(model,
                                                        X_lower,
                                                        X_upper,
                                                        Nb=n_representer)

    def update(self, model, cost_model, overhead=None):
        self.cost_model = cost_model
        if overhead is None:
            self.overhead = 0
        else:
            self.overhead = overhead
        super(InformationGainPerUnitCost, self).update(model)

    def compute(self, X, derivative=False):
        """
        Computes the acquisition value for a single point.

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
            # dh, g = super(EnvironmentEntropy, self).compute(X,
            #                                    derivative=derivative)

            # dmu = self.cost_model.predictive_gradients(
            # X[:, self.is_env_variable == 1])[0]
            # log_cost = (log_cost + 1e-8)
            # acquisition_value = dh / log_cost
            # grad = g * log_cost + dmu * dh

            # return acquisition_value, grad
            raise("Not implemented")
        else:
            dh = super(InformationGainPerUnitCost, self).compute(X,
                                            derivative=derivative)
            # We model the log cost, but we compute
            # the information gain per unit cost

            # Add the cost it took to pick the last configuration
            cost = np.exp(log_cost) + self.overhead

            acquisition_value = dh / cost

            return acquisition_value

    def sampling_acquisition_wrapper(self, x):

        # Check if sample point is inside the configuration space
        X_lower = self.X_lower[np.where(self.is_env == 0)]
        X_upper = self.X_upper[np.where(self.is_env == 0)]
        if np.any(x < X_lower) or np.any(x > X_upper):
            return -np.inf

        # Project point to subspace
        proj_x = np.concatenate((x, self.X_upper[self.is_env == 1]))
        return self.sampling_acquisition(np.array([proj_x]))[0]

    def sample_representer_points(self):
        # Sample representer points only in the
        # configuration space by setting all environmental
        # variables to 1
        D = np.where(self.is_env == 0)[0].shape[0]

        X_lower = self.X_lower[np.where(self.is_env == 0)]
        X_upper = self.X_upper[np.where(self.is_env == 0)]

        self.sampling_acquisition.update(self.model)

        restarts = np.random.uniform(low=X_lower,
                                     high=X_upper,
                                     size=(self.Nb, D))

        sampler = emcee.EnsembleSampler(self.Nb, D,
                                    self.sampling_acquisition_wrapper)

        self.zb, self.lmb, _ = sampler.run_mcmc(restarts, 20)

        if len(self.zb.shape) == 1:
            self.zb = self.zb[:, None]
        if len(self.lmb.shape) == 1:
            self.lmb = self.lmb[:, None]

        # Project representer points to subspace
        proj = np.ones([self.zb.shape[0],
                    self.X_upper[self.is_env == 1].shape[0]])
        proj *= self.X_upper[self.is_env == 1].shape[0]
        self.zb = np.concatenate((self.zb, proj), axis=1)
