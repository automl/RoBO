import numpy as np

from scipy import optimize

from robo.initial_design.init_random_uniform import init_random_uniform


def posterior_mean_optimization(model, lower, upper, n_restarts=10, with_gradients=False):
    """
    Estimates the incumbent by minimize the posterior
    mean of the objective function.

    Parameters
    ----------
    model : Model object
        Posterior belief of the objective function.
    lower : (D) numpy array
        Specified the lower bound of the input space. Each entry
        corresponds to one dimension.
    upper : (D) numpy array
        Specified the upper bound of the input space. Each entry
        corresponds to one dimension.
    n_restarts: int
        Number of independent restarts of the optimization procedure from random starting points
    with_gradients : bool
        Specifies if gradient information are used. Only valid
        if method == 'scipy'.

    Returns
    -------
    np.ndarray(D,)
        best point that was found
    """

    def f(x):
        return model.predict(x[np.newaxis, :])[0][0]

    def df(x):
        dmu = model.predictive_gradients(x[np.newaxis, :])[0]
        return dmu

    startpoints = init_random_uniform(lower, upper, n_restarts)

    x_opt = np.zeros([len(startpoints), lower.shape[0]])
    fval = np.zeros([len(startpoints)])
    for i, startpoint in enumerate(startpoints):
        if with_gradients:
            res = optimize.fmin_l_bfgs_b(f, startpoint, df, bounds=list(zip(lower, upper)))
            x_opt[i] = res[0]
            fval[i] = res[1]
        else:
            res = optimize.minimize(f, startpoint, bounds=list(zip(lower, upper)), method="L-BFGS-B")
            x_opt[i] = res["x"]
            fval[i] = res["fun"]

    # Return the point with the lowest function value
    best = np.argmin(fval)
    return x_opt[best]


def posterior_mean_plus_std_optimization(model, lower, upper, n_restarts=10, with_gradients=False):
    """
    Estimates the incumbent by minimize the posterior mean + std of the objective function, i.e. the
    upper bound.

    Parameters
    ----------
    model : Model object
        Posterior belief of the objective function.
    lower : (D) numpy array
        Specified the lower bound of the input space. Each entry
        corresponds to one dimension.
    upper : (D) numpy array
        Specified the upper bound of the input space. Each entry
        corresponds to one dimension.
    n_restarts: int
        Number of independent restarts of the optimization procedure from random starting points
    with_gradients : bool
        Specifies if gradient information are used. Only valid
        if method == 'scipy'.

    Returns
    -------
    np.ndarray(D,)
        best point that was found
    """

    def f(x):
        mu, var = model.predict(x[np.newaxis, :])
        return (mu + np.sqrt(var))[0]

    def df(x):
        dmu, dvar = model.predictive_gradients(x[np.newaxis, :])
        _, var = model.predict(x[np.newaxis, :])
        std = np.sqrt(var)
        # To get the gradients of the standard deviation
        # We need to apply chain rule
        # (s(x)=sqrt[v(x)] => s'(x) = 1/2 * v'(x) / sqrt[v(x)]
        dstd = 0.5 * dvar / std
        return dmu[:, :, 0] + dstd

    startpoints = init_random_uniform(lower, upper, n_restarts)

    x_opt = np.zeros([len(startpoints), lower.shape[0]])
    fval = np.zeros([len(startpoints)])
    for i, startpoint in enumerate(startpoints):
        if with_gradients:
            res = optimize.fmin_l_bfgs_b(f, startpoint, df, bounds=list(zip(lower, upper)))
            x_opt[i] = res[0]
            fval[i] = res[1]
        else:
            res = optimize.minimize(f, startpoint, bounds=list(zip(lower, upper)), method="L-BFGS-B")
            x_opt[i] = res["x"]
            fval[i] = res["fun"]

    # Return the point with the lowest function value
    best = np.argmin(fval)
    return x_opt[best]
