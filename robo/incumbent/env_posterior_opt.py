'''
Created on Dec 16, 2015

@author: Aaron Klein
'''


import cma
import numpy as np
from scipy import optimize

from robo.incumbent.incumbent_estimation import IncumbentEstimation


class EnvPosteriorMeanOptimization(IncumbentEstimation):

    def __init__(self, model, X_lower, X_upper, is_env,
                 method="scipy", with_gradients=False):
        """
        Estimates the incumbent by minimize the current posterior
        mean of the objective function in the configuration subspace.

        Parameters
        ----------
        model : Model object
            Models the objective function.
        X_lower : (D) numpy array
            Specified the lower bound of the input space. Each entry
            corresponds to one dimension.
        X_upper : (D) numpy array
            Specified the upper bound of the input space. Each entry
            corresponds to one dimension.
        is_env : (D) numpy array
            Specified if the corresponding dimensions is an environmental
            variable (1) or not (0)
        method : ['scipy', 'cmaes']
            Specifies which optimization method is used to minimize
            the posterior mean.
        with_gradients : bool
            Specifies if gradient information are used. Only valid
            if method == 'scipy'.
        """

        super(EnvPosteriorMeanOptimization, self).__init__(model,
                                                        X_lower,
                                                        X_upper)
        self.is_env = is_env
        self.env_values = X_upper[is_env == 1]
        self.sub_X_lower = X_lower[is_env == 0]
        self.sub_X_upper = X_upper[is_env == 0]
        self.method = method
        self.with_gradients = with_gradients

    def f(self, x):
        # Project x to the subspace
        x_ = np.zeros([self.is_env.shape[0]])
        x_[self.is_env == 1] = self.env_values
        x_[self.is_env == 0] = x

        mu = self.model.predict(x_[np.newaxis, :])[0]

        return mu[0, 0]

    def df(self, x):
        # Project x to the subspace
        x_ = np.zeros([self.is_env.shape[0]])
        x_[self.is_env == 1] = self.env_values
        x_[self.is_env == 0] = x

        # Get gradients and variance of the test point
        dmu = self.model.predictive_gradients(x_[np.newaxis, :])[0]
        # Return gradients of the dimensions of the projected
        # subspace (discard the others)
        return dmu[:, :, 0][0, self.is_env == 0]

    def estimate_incumbent(self, startpoints):
        """
        Starts form each startpoint an optimization run in the configuration
        subspace and returns the best found point of all runs as incumbent

        Parameters
        ----------
        startpoints : (N, D) numpy array
            Startpoints where the optimization starts from.

        Returns
        -------
        np.ndarray(1, D)
            Incumbent
        np.ndarray(1,1)
            Incumbent value
        """
        x_opt = np.zeros([len(startpoints), self.X_lower.shape[0]])
        fval = np.zeros([len(startpoints)])
        for i, startpoint in enumerate(startpoints):
            if self.method == "scipy":
                if self.with_gradients:
                    res = optimize.fmin_l_bfgs_b(self.f,
                                startpoint[self.is_env == 0],
                                self.df,
                                bounds=list(zip(self.sub_X_lower, self.sub_X_upper)))
                    # The result has the dimensionality of the projected
                    # configuration space so we have to add the
                    # dimensions of the environmental subspace
                    x_ = np.zeros([self.is_env.shape[0]])
                    x_[self.is_env == 1] = self.env_values
                    x_[self.is_env == 0] = res[0]
                    x_opt[i] = x_
                    fval[i] = res[1]
                else:
                    res = optimize.minimize(self.f,
                        startpoint[self.is_env == 0],
                        bounds=list(zip(self.sub_X_lower, self.sub_X_upper)),
                        method="L-BFGS-B",
                        options={"disp": True})
                    x_ = np.zeros([self.is_env.shape[0]])
                    x_[self.is_env == 1] = self.env_values
                    x_[self.is_env == 0] = res["x"]
                    x_opt[i] = x_
                    fval[i] = res["fun"]
            elif self.method == 'cmaes':
                res = cma.fmin(self.f, startpoint[self.is_env == 0], 0.6,
                    options={"bounds": [self.sub_X_lower, self.sub_X_upper]})
                x_ = np.zeros([self.is_env.shape[0]])
                x_[self.is_env == 1] = self.env_values
                x_[self.is_env == 0] = res[0]
                x_opt[i] = x_
                fval[i] = res[1]
        # Return the point with the lowest function value
        best = np.argmin(fval)
        return x_opt[best, np.newaxis, :], np.array([[fval[best]]])


class EnvPosteriorMeanAndStdOptimization(EnvPosteriorMeanOptimization):

    def __init__(self, model, X_lower, X_upper, is_env,
                 method="scipy", with_gradients=False):
        """
        Estimates the incumbent by minimize the current posterior
        mean + std of the objective function in the configuration
        subspace.

        Parameters
        ----------
        model : Model object
            Models the objective function.
        X_lower : (D) numpy array
            Specified the lower bound of the input space. Each entry
            corresponds to one dimension.
        X_upper : (D) numpy array
            Specified the upper bound of the input space. Each entry
            corresponds to one dimension.
        is_env : (D) numpy array
            Specified if the corresponding dimensions is an environmental
            variable (1) or not (0)
        method : ['scipy', 'cmaes']
            Specifies which optimization method is used to minimize
            the posterior mean.
        with_gradients : bool
            Specifies if gradient information are used. Only valid
            if method == 'scipy'.
        """
        super(EnvPosteriorMeanAndStdOptimization, self).__init__(model,
                                                        X_lower,
                                                        X_upper,
                                                        is_env,
                                                        method,
                                                        with_gradients)

    def f(self, x):
        # Project x to the subspace
        x_ = np.zeros([self.is_env.shape[0]])
        x_[self.is_env == 1] = self.env_values
        x_[self.is_env == 0] = x

        mu, var = self.model.predict(x_[np.newaxis, :])

        return (mu + np.sqrt(var))[0, 0]

    def df(self, x):
        # Project x to the subspace
        x_ = np.zeros([self.is_env.shape[0]])
        x_[self.is_env == 1] = self.env_values
        x_[self.is_env == 0] = x

        # Get gradients and variance of the test point
        dmu, dvar = self.model.predictive_gradients(x_[np.newaxis, :])
        _, var = self.model.predict(x_[np.newaxis, :])

        # To get the gradients of the standard deviation
        # We need to apply chain rule
        # (s(x)=sqrt[v(x)] => s'(x) = 1/2 * v'(x) / sqrt[v(x)]
        std = np.sqrt(var)
        dstd = 0.5 * dvar / std

        # Return gradients of the dimensions of the projected
        # subspace (discard the others)
        return (dmu[:, :, 0] + dstd)[0, self.is_env == 0]
