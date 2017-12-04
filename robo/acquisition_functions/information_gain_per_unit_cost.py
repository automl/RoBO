import emcee
import numpy as np

from robo.acquisition_functions.information_gain import InformationGain


class InformationGainPerUnitCost(InformationGain):

    def __init__(self, model, cost_model,
                 lower, upper,
                 is_env_variable,
                 sampling_acquisition=None,
                 n_representer=50):
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
            Gaussian process.
        cost_model : model
            Models the cost function. The model has to be a Gaussian Process.
        lower : (D) numpy array
            Specified the lower bound of the input space. Each entry
            corresponds to one dimension.
        upper : (D) numpy array
            Specified the upper bound of the input space. Each entry
            corresponds to one dimension.
        is_env_variable : (D) numpy array
            Specifies which input dimension is an environmental variable. If
            the i-th input is an environmental variable than the i-th entry has
            to be 1 and 0 otherwise.
        n_representer : int, optional
            The number of representer points to discretize the input space and
            to compute pmin.
        """
        self.cost_model = cost_model
        self.n_dims = lower.shape[0]

        self.is_env = is_env_variable

        super(InformationGainPerUnitCost, self).__init__(model,
                                                         lower,
                                                         upper,
                                                         sampling_acquisition=sampling_acquisition,
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
        Computes the acquisition_functions value for a single point.

        Parameters
        ----------
        X : (1, D) numpy array
            The input point for which the acquisition_functions functions is computed.
        derivative : bool, optional
            If it is equal to True also the derivatives with respect to X is
            computed.

        Returns
        -------
        acquisition_value: numpy array
            The acquisition_functions value computed for X.
        grad : numpy array
            The computed gradient of the acquisition_functions function at X. Only
            returned if derivative==True
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        # Predict the log costs for this configuration
        log_cost = self.cost_model.predict(X)[0]

        if derivative:
            raise "Not implemented"
        else:
            dh = super(InformationGainPerUnitCost, self).compute(X,
                                                                 derivative=derivative)
            # We model the log cost, but we compute
            # the information gain per unit cost

            # Add the cost it took to pick the last configuration
            cost = np.exp(log_cost)

            acquisition_value = dh / (cost + self.overhead)

            return acquisition_value

    def sampling_acquisition_wrapper(self, x):

        # Check if sample point is inside the configuration space
        lower = self.lower[np.where(self.is_env == 0)]
        upper = self.upper[np.where(self.is_env == 0)]
        if np.any(x < lower) or np.any(x > upper):
            return -np.inf

        # Project point to subspace
        proj_x = np.concatenate((x, self.upper[self.is_env == 1]))
        return self.sampling_acquisition(np.array([proj_x]))[0]

    def sample_representer_points(self):
        # Sample representer points only in the
        # configuration space by setting all environmental
        # variables to 1
        D = np.where(self.is_env == 0)[0].shape[0]

        lower = self.lower[np.where(self.is_env == 0)]
        upper = self.upper[np.where(self.is_env == 0)]

        self.sampling_acquisition.update(self.model)

        for i in range(5):
            restarts = np.random.uniform(low=lower,
                                         high=upper,
                                         size=(self.Nb, D))
            sampler = emcee.EnsembleSampler(self.Nb, D,
                                        self.sampling_acquisition_wrapper)

            self.zb, self.lmb, _ = sampler.run_mcmc(restarts, 50)
            if not np.any(np.isinf(self.lmb)):
                break
            else:
                print("Infinity")
        if np.any(np.isinf(self.lmb)):
            raise ValueError("Could not sample valid representer points! LogEI is -infinity")
        if len(self.zb.shape) == 1:
            self.zb = self.zb[:, None]
        if len(self.lmb.shape) == 1:
            self.lmb = self.lmb[:, None]

        # Project representer points to subspace
        proj = np.ones([self.zb.shape[0],
                    self.upper[self.is_env == 1].shape[0]])
        proj *= self.upper[self.is_env == 1].shape[0]
        self.zb = np.concatenate((self.zb, proj), axis=1)
