'''
Created on Oct 12, 2015

@author: Aaron Klein
'''

import logging
import george
import numpy as np

from scipy import optimize
from copy import deepcopy

from robo.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class GaussianProcess(BaseModel):

    def __init__(self, kernel, prior=None,
                 yerr=1e-25, use_gradients=True,
                 basis_func=None, dim=None, *args, **kwargs):
        """
        Interface to the george GP library. The GP hyperparameter are obtained
        by optimizing the marginal loglikelihood.

        Parameters
        ----------
        kernel : george kernel object
            Specifies the kernel that is used for all Gaussian Process
        prior : prior object
            Defines a prior for the hyperparameters of the GP. Make sure that
            it implements the Prior interface.
        yerr : float
            Noise term that is added to the diagonal of the covariance matrix
            for the cholesky decomposition.
        use_gradients : bool
            Use gradient information to optimize the negative log likelihood
        """

        self.kernel = kernel
        self.model = None
        self.prior = prior
        self.yerr = yerr
        self.use_gradients = use_gradients
        self.basis_func = basis_func
        self.dim = dim

    def scale(self, x, new_min, new_max, old_min, old_max):
        return ((new_max - new_min) *
                (x - old_min) / (old_max - old_min)) + new_min

    def train(self, X, Y, do_optimize=True):
        """
        Computes the cholesky decomposition of the covariance of X and
        estimates the GP hyperparameter by optimizing the marginal
        loglikelihood. The prior mean of the GP is set to the empirical
        mean of the X.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        Y: np.ndarray (N, 1)
            The corresponding target values.
        do_optimize: boolean
            If set to true the hyperparameters are optimized.
        """
        self.X = X
        # For EnvES we transform s to (1 - s)^2
        if self.basis_func is not None:
            self.X = deepcopy(X)
            self.X[:, self.dim] = self.basis_func(self.X[:, self.dim])
        self.Y = Y

        # Use the mean of the data as mean for the GP
        self.mean = np.mean(Y, axis=0)
        self.model = george.GP(self.kernel, mean=self.mean)

        # Precompute the covariance
        while(True):
            try:
                self.model.compute(self.X, yerr=self.yerr)
                break
            except np.linalg.LinAlgError:
                if self.yerr == 0.0:
                    self.yerr = 1e-25
                else:
                    self.yerr *= 10
                logger.error(
                    "Cholesky decomposition for the covariance matrix \
                    of the GP failed. \
                    Add %s noise on the diagonal." %
                    self.yerr)

        if do_optimize:
            self.hypers = self.optimize()
            logger.info("HYPERS: " + str(self.hypers))
            self.model.kernel[:] = self.hypers
        else:
            self.hypers = self.model.kernel[:]

    def get_noise(self):
        # Assumes a kernel of the form amp * (kernel1 + noise_kernel)
        # FIXME: How to determine the noise of george gp in general?
        return self.yerr
        assert self.kernel.k2.k2.kernel_type == 1
        return self.kernel.k2.k2.pars[0]

    def nll(self, theta):
        """
        Returns the negative marginal log likelihood (+ the prior) for
        a hyperparameter configuration theta.
        (negative because we use scipy minimize for optimization)

        Parameters
        ----------
        theta : np.ndarray(H)
            Hyperparameter vector. Note that all hyperparameter are
            on a log scale.

        Returns
        ----------
        float
            lnlikelihood + prior
        """

        # Specify bounds to keep things sane
        if np.any((-10 > theta) + (theta > 10)):
            return 1e25

        self.model.kernel[:] = theta
        ll = self.model.lnlikelihood(self.Y[:, 0], quiet=True)

        # Add prior
        if self.prior is not None:
            ll += self.prior.lnprob(theta)

        # We add a minus here because scipy is minimizing
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(self, theta):
        self.model.kernel[:] = theta

        gll = self.model.grad_lnlikelihood(self.Y[:, 0], quiet=True)

        if self.prior is not None:
            gll += self.prior.gradient(theta)
        return -gll

    def optimize(self):
        # Start optimization  from the previous hyperparameter configuration
        p0 = self.model.kernel.vector
        if self.use_gradients:
            results = optimize.minimize(self.nll, p0, jac=self.grad_nll)
        else:
            results = optimize.minimize(self.nll, p0)
        return results.x

    def predict_variance(self, X1, X2):
        r"""
        Predicts the variance between two test points X1, X2 by
           math: \sigma(X_1, X_2) = k_{X_1,X_2} - k_{X_1,X} * (K_{X,X}
                       + \sigma^2*\mathds{I})^-1 * k_{X,X_2})

        Parameters
        ----------
        X1: np.ndarray (N, D)
            First test point
        X2: np.ndarray (N, D)
            Second test point
        Returns
        ----------
        np.array(N,1)
            predictive variance

        """
        # For EnvES we transform s to (1 - s)^2
        if self.basis_func is not None:
            X_test_1 = deepcopy(X1)
            X_test_1[:, self.dim] = self.basis_func(X_test_1[:, self.dim])
            X_test_2 = deepcopy(X2)
            X_test_2[:, self.dim] = self.basis_func(X_test_2[:, self.dim])
        else:
            X_test_1 = X1
            X_test_2 = X2
        x_ = np.concatenate((X_test_1, X_test_2))
        _, var = self.predict(x_)
        var = var[:-1, -1, np.newaxis]

        return var

    def predict(self, X, **kwargs):
        r"""
        Returns the predictive mean and variance of the objective function at
        the specified test point.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input test points

        Returns
        ----------
        np.array(N,1)
            predictive mean
        np.array(N,1)
            predictive variance

        """
        # For EnvES we transform s to (1 - s)^2
        if self.basis_func is not None:
            X_test = deepcopy(X)
            X_test[:, self.dim] = self.basis_func(X_test[:, self.dim])
        else:
            X_test = X

        if self.model is None:
            logger.error("The model has to be trained first!")
            raise ValueError

        mu, var = self.model.predict(self.Y[:, 0], X_test)

        # Clip negative variances and set them to the smallest
        # positive float values
        if var.shape[0] == 1:
            var = np.clip(var, np.finfo(var.dtype).eps, np.inf)
        else:
            var[np.diag_indices(var.shape[0])] = np.clip(
                                            var[np.diag_indices(var.shape[0])],
                                            np.finfo(var.dtype).eps, np.inf)
            var[np.where((var < np.finfo(var.dtype).eps) & (var > -np.finfo(var.dtype).eps))] = 0

        return mu[:, np.newaxis], var

    def sample_functions(self, X_test, n_funcs=1):
        """
        Samples F function values from the current posterior at the N
        specified test point.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        n_funcs: int
            Number of function values that are drawn at each test point.

        Returns
        ----------
        np.array(F,N)
            The F function values drawn at the N test points.
        """

        return self.model.sample_conditional(self.Y[:, 0], X_test, n_funcs)

    def predictive_gradients(self, X_test):
        dmdx, dvdx = self.m.predictive_gradients(Xnew)
        return dmdx[:, 0, :], dvdx