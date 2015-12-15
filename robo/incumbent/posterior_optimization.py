'''
Created on 14.07.2015

@author: Aaron Klein
'''

import numpy as np
from scipy import optimize

from robo.incumbent.incumbent_estimation import IncumbentEstimation


class PosteriorMeanOptimization(IncumbentEstimation):
    
    def __init__(self, model, X_lower, X_upper,
                 method="scipy", with_gradients=False):
        """
        Estimates the incumbent by minimize the current posterior
	mean of the objective function.

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
	method : ['scipy', 'cmaes']
	    Specifies which optimization method is used to minimize
	    the posterior mean.
	with_gradients : bool
	    Specfies if gradient information are used. Only valid
	    if method == 'scipy'.
        """
        

	super(PosteriorMeanOptimization, self).__init__(model, X_lower, X_upper)
        self.method = method
        self.with_gradients = with_gradients

    def f(self, x):
        return self.model.predict(x[np.newaxis, :])[0]

    def df(self, x):
        dmu = self.model.predictive_gradients(x[np.newaxis, :])[0]
        return dmu[0, :, :]

    def estimate_incumbent(self, startpoints):
	"""
        Starts form each startpoint an optimization run and returns
	the best found point of all runs as incumbent

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
                    res = optimize.fmin_l_bfgs_b(self.f, startpoint, self.df,
                                    bounds=list(zip(self.X_lower, self.X_upper)))
                    x_opt[i] = res[0]
                    fval[i] = res[1]
                else:
                    res = optimize.minimize(self.f, startpoint,
                                bounds=list(zip(self.X_lower, self.X_upper)),
                                method="L-BFGS-B")
                    x_opt[i] = res["x"]
                    fval[i] = res["fun"]
            elif self.method == 'cmaes':
                res = cma.fmin(self.f, startpoint, 0.6,
                              options={"bounds": [self.X_lower, self.X_upper]})
                x_opt[i] = res[0]
                fval[i] = res[1]

        # Return the point with the lowest function value
        best = np.argmin(fval)
        return x_opt[best, np.newaxis, :], fval[best, np.newaxis, np.newaxis]


class PosteriorMeanAndStdOptimization(PosteriorMeanOptimization):
    
    def __init__(self, model, X_lower, X_upper,
                 method="scipy", with_gradients=False):
        """
        Estimates the incumbent by minimize the current posterior
	mean + std of the objective function.

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
	method : ['scipy', 'cmaes']
	    Specifies which optimization method is used to minimize
	    the posterior mean.
	with_gradients : bool
	    Specfies if gradient information are used. Only valid
	    if method == 'scipy'.
        """
	super(PosteriorMeanAndStdOptimization, self).__init__(model,
                                                        X_lower,
                                                        X_upper,
                                                        method,
                                                        with_gradients)
    def f(self, x):
        mu, var = self.model.predict(x[np.newaxis, :])
        return (mu + np.sqrt(var))

    def df(self, x):
        dmu, dvar = self.model.predictive_gradients(x[np.newaxis, :])
        _, var = self.model.predict(x[np.newaxis, :])
        std = np.sqrt(var)
        # To get the gradients of the standard deviation
        # We need to apply chain rule (s(x)=sqrt[v(x)] => s'(x) = 1/2 * v'(x) / sqrt[v(x)]
        dstd = 0.5 * dvar / std
        return (dmu[:, :, 0] + dstd)[0, :]



def env_optimize_posterior_mean_and_std(
        model,
        X_lower,
        X_upper,
        is_env,
        startpoints,
        with_gradients=True):

    # We only optimize the posterior in the projected subspace
    env_values = X_upper[is_env == 1]
    sub_X_lower = X_lower[is_env == 0]
    sub_X_upper = X_upper[is_env == 0]

    def f(x):
        # Project x to the subspace
        x_ = np.zeros([is_env.shape[0]])
        x_[is_env == 1] = env_values
        x_[is_env == 0] = x

        mu, var = model.predict(x_[np.newaxis, :])

        return (mu + np.sqrt(var))

    def df(x):
        # Project x to the subspace
        x_ = np.zeros([is_env.shape[0]])
        x_[is_env == 1] = env_values
        x_[is_env == 0] = x

        # Get gradients and variance of the test point
        dmu, dvar = model.predictive_gradients(x_[np.newaxis, :])
        _, var = model.predict(x_[np.newaxis, :])

        # To get the gradients of the standard deviation
        # We need to apply chain rule (s(x)=sqrt[v(x)] => s'(x) = 1/2 * v'(x) / sqrt[v(x)]
        std = np.sqrt(var)
        dstd = 0.5 * dvar / std

        # Return gradients of the dimensions of the projected subspace (discard the others)
        return (dmu[:, :, 0] + dstd)[0, is_env == 0]

    x_opt = np.zeros([len(startpoints), X_lower.shape[0]])
    fval = np.zeros([len(startpoints)])
    for i, startpoint in enumerate(startpoints):

        if with_gradients:
            res = optimize.fmin_l_bfgs_b(
                f, startpoint[
                    is_env == 0], df, bounds=zip(
                    sub_X_lower, sub_X_upper))
            # The result has the dimensionality of the projected configuration space
            # so we have to add the dimensions of the environmental subspace
            x_ = np.zeros([is_env.shape[0]])
            x_[is_env == 1] = env_values
            x_[is_env == 0] = res[0]
            x_opt[i] = x_
            fval[i] = res[1]
        else:
            res = optimize.minimize(
                f,
                startpoint[
                    is_env == 0],
                bounds=zip(
                    sub_X_lower,
                    sub_X_upper),
                method="L-BFGS-B",
                options={
                    "disp": True})
            x_ = np.zeros([is_env.shape[0]])
            x_[is_env == 1] = env_values
            x_[is_env == 0] = res["x"]
            x_opt[i] = x_
            fval[i] = res["fun"]

    # Return the point with the lowest function value
    best = np.argmin(fval)
    return x_opt[best], fval[best]


def env_optimize_posterior_mean_and_std_mcmc(
        model,
        X_lower,
        X_upper,
        is_env,
        startpoint,
        with_gradients=False):
    # If we perform MCMC over the model's hyperparameter we optimize
    # each model individually and return the best point we found
    # TODO: I think it might be better if we optimize the averaged posterior instead
    incumbents = np.zeros([len(model.models), startpoint.shape[1]])
    vals = np.zeros([len(model.models)])
    for i, m in enumerate(model.models):
        inc, inc_val = env_optimize_posterior_mean_and_std(
            m, X_lower, X_upper, is_env, startpoint, with_gradients)
        incumbents[i] = inc
        vals[i] = inc_val

    idx = np.argmin(incumbents)
    incumbent_value = vals[idx]
    incumbent = incumbents[idx]
    return incumbent, incumbent_value
