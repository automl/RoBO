'''
Created on 14.07.2015

@author: Aaron Klein
'''

from scipy import optimize
import numpy as np
from robo.recommendation.incumbent import compute_incumbent


def optimize_posterior_mean(model, X_lower, X_upper, inc=None, with_gradients=False):
    def f(x):
        return model.predict(x[np.newaxis, :])[0]

    def df(x):
        dmu = model.predictive_gradients(x[np.newaxis, :])[0]
        return dmu[0, :, :]

    if inc is None:
        inc, _ = compute_incumbent(model)

    if with_gradients:
        res = optimize.minimize(f, inc, bounds=zip(X_lower, X_upper), jac=df)
    else:
        res = optimize.minimize(f, inc, bounds=zip(X_lower, X_upper))
    return res["x"], res["fun"]


def optimize_posterior_mean_and_std(model, X_lower, X_upper, startpoints=None, with_gradients=True):
    def f(x):
        mu, var = model.predict(x[np.newaxis, :])
        return (mu + np.sqrt(var))

    def df(x):
        dmu, dvar = model.predictive_gradients(x[np.newaxis, :])
        _, var = model.predict(x[np.newaxis, :])
        std = np.sqrt(var)
        # To get the gradients of the standard deviation
        # We need to apply chain rule (s(x)=sqrt[v(x)] => s'(x) = 1/2 * v'(x) / sqrt[v(x)]
        dstd = 0.5 * dvar / std
        return (dmu[:, :, 0] + dstd)[0, :]

    if startpoints is None:
        startpoints = []
        startpoints.append(compute_incumbent(model)[0])

    x_opt = np.zeros([len(startpoints), X_lower.shape[0]])
    fval = np.zeros([len(startpoints)])
    for i, startpoint in enumerate(startpoints):
        if with_gradients:
            res = optimize.fmin_l_bfgs_b(f, startpoint, df, bounds=zip(X_lower, X_upper))
            x_opt[i] = res[0]
            fval[i] = res[1]
        else:
            res = optimize.minimize(f, startpoint, bounds=zip(X_lower, X_upper), method="L-BFGS-B")
            x_opt[i] = res["x"]
            fval[i] = res["fun"]

    # Return the point with the lowest function value
    best = np.argmin(fval)
    return x_opt[best], fval[best]


def env_optimize_posterior_mean_and_std(model, X_lower, X_upper, is_env, startpoint, with_gradients=False):

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

    if with_gradients:
        res = optimize.fmin_l_bfgs_b(f, startpoint, df, bounds=zip(sub_X_lower, sub_X_upper))
        # The result has the dimensionality of the projected subspace add the dimensions of the environmental subspace
        x_ = np.zeros([is_env.shape[0]])
        x_[is_env == 1] = env_values
        x_[is_env == 0] = res[0]
        return x_, res[1]
    else:
        res = optimize.minimize(f, startpoint[:, is_env == 0], bounds=zip(sub_X_lower, sub_X_upper), method="L-BFGS-B", options={"disp": True})
        # The result has the dimensionality of the projected subspace add the dimensions of the environmental subspace
        x_ = np.zeros([is_env.shape[0]])
        x_[is_env == 1] = env_values
        x_[is_env == 0] = res["x"]
        return x_, res["fun"]


def env_optimize_posterior_mean_and_std_mcmc(model, X_lower, X_upper, is_env, startpoint, with_gradients=False):
    # If we perform MCMC over the model's hyperparameter we optimize each model individually and return the best point we found
    # TODO: I think it might be better if we optimize the averaged posterior instead
    incumbents = np.zeros([len(model.models), startpoint.shape[1]])
    vals = np.zeros([len(model.models)])
    for i, m in enumerate(model.models):
        inc, inc_val = env_optimize_posterior_mean_and_std(m, X_lower, X_upper, is_env, startpoint, with_gradients)
        incumbents[i] = inc
        vals[i] = inc_val

    idx = np.argmin(incumbents)
    incumbent_value = vals[idx]
    incumbent = incumbents[idx]
    return incumbent, incumbent_value

