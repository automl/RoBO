'''
Created on 14.07.2015

@author: Aaron Klein
'''
import cma
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
            Specifies if gradient information are used. Only valid
            if method == 'scipy'.
        """

        super(PosteriorMeanOptimization, self).__init__(model,
                                                        X_lower,
                                                        X_upper)
        self.method = method
        self.with_gradients = with_gradients

    def f(self, x):
        return self.model.predict(x[np.newaxis, :])[0][0][0]

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
            Specifies if gradient information are used. Only valid
            if method == 'scipy'.
        """
        super(PosteriorMeanAndStdOptimization, self).__init__(model,
                                                        X_lower,
                                                        X_upper,
                                                        method,
                                                        with_gradients)

    def f(self, x):
        mu, var = self.model.predict(x[np.newaxis, :])
        return (mu + np.sqrt(var))[0, 0]

    def df(self, x):
        dmu, dvar = self.model.predictive_gradients(x[np.newaxis, :])
        _, var = self.model.predict(x[np.newaxis, :])
        std = np.sqrt(var)
        # To get the gradients of the standard deviation
        # We need to apply chain rule
        # (s(x)=sqrt[v(x)] => s'(x) = 1/2 * v'(x) / sqrt[v(x)]
        dstd = 0.5 * dvar / std
        return (dmu[:, :, 0] + dstd)[0, :]
