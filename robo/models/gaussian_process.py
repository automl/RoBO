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
                 noise=1e-3, use_gradients=False,
                 basis_func=None, dim=None, normalize_output=False,
                 *args, **kwargs):
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
        noise : float
            Noise term that is added to the diagonal of the covariance matrix
            for the cholesky decomposition.
        use_gradients : bool
            Use gradient information to optimize the negative log likelihood
        """

        self.kernel = kernel
        self.model = None
        self.prior = prior
        self.noise = noise
        self.use_gradients = use_gradients
        self.basis_func = basis_func
        self.dim = dim
        self.normalize_output = normalize_output
        self.X = None
        self.Y = None

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
        # For Fabolas we transform s to (1 - s)^2
        if self.basis_func is not None:
            self.X = deepcopy(X)
            self.X[:, self.dim] = self.basis_func(self.X[:, self.dim])

        self.Y = Y
        if self.normalize_output:
            self.Y_mean = np.mean(Y)
            self.Y_std = np.std(Y)
            self.Y = (Y - self.Y_mean) / self.Y_std

        # Use the empirical mean of the data as mean for the GP
        self.mean = np.mean(self.Y, axis=0)
        self.model = george.GP(self.kernel, mean=self.mean)

        if do_optimize:
            self.hypers = self.optimize()
            self.model.kernel[:] = self.hypers[:-1]
            self.noise = np.exp(self.hypers[-1]) ## sigma^2
        else:
            self.hypers = self.model.kernel[:]
            self.hypers = np.append(self.hypers, np.log(self.noise))
        logger.debug("HYPERS: " + str(self.hypers))
        self.model.compute(self.X, yerr=np.sqrt(self.noise))

    def get_noise(self):
        return self.noise

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
        if np.any((-20 > theta) + (theta > 20)):
            return 1e25

        # The last entry of theta is always the noise
        self.model.kernel[:] = theta[:-1]
        noise = np.exp(theta[-1])  # sigma^2
        
        self.model.compute(self.X, yerr=np.sqrt(noise))
        ll = self.model.lnlikelihood(self.Y[:, 0], quiet=True)

        # Add prior
        if self.prior is not None:
            ll += self.prior.lnprob(theta)

        # We add a minus here because scipy is minimizing
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(self, theta):

        self.model.kernel[:] = theta[:-1]
        noise = np.exp(theta[-1])
        
        self.model.compute(self.X, yerr=np.sqrt(noise))
        
        self.model._compute_alpha(self.Y[:, 0])
        K_inv = self.model.solver.apply_inverse(np.eye(self.model._alpha.size),
                                          in_place=True)

        # The gradients of the Gram matrix, for the noise this is just 
        # the identiy matrix
        Kg = self.model.kernel.gradient(self.model._x)
        Kg = np.concatenate((Kg, np.eye(Kg.shape[0])[:, :, None]), axis=2)

        # Calculate the gradient.
        A = np.outer(self.model._alpha, self.model._alpha) - K_inv
        g = 0.5 * np.einsum('ijk,ij', Kg, A)
        

        if self.prior is not None:
            g += self.prior.gradient(theta)

        return -g

    def optimize(self):
        # Start optimization  from the previous hyperparameter configuration
        p0 = self.model.kernel.vector
        p0 = np.append(p0, np.log(self.noise))

        if self.use_gradients:
            bounds = [(-10, 10)] * (len(self.kernel) + 1)
            theta, _, _ = optimize.fmin_l_bfgs_b(self.nll, p0,
                                             fprime=self.grad_nll,
                                             bounds=bounds)
        else:
            results = optimize.minimize(self.nll, p0)
            theta = results.x
        return theta

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
        x_ = np.concatenate((X1, X2))
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

        # For Fabolas we transform s to (1 - s)^2
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
        dmdx, dvdx = self.m.predictive_gradients(X_test)
        return dmdx[:, 0, :], dvdx