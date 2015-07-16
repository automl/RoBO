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


def optimize_posterior_mean_and_std(model, X_lower, X_upper, inc=None, with_gradients=False):
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
        return (dmu + dstd)

    if inc is None:
        inc, _ = compute_incumbent(model)

    if with_gradients:
        res = optimize.minimize(f, inc, bounds=zip(X_lower, X_upper), jac=df)
    else:
        res = optimize.minimize(f, inc, bounds=zip(X_lower, X_upper))

    return res["x"], res["fun"]
