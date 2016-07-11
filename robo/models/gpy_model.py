
import numpy as np
import GPy
import logging

from robo.models.base_model import BaseModel
from copy import deepcopy

logger = logging.getLogger(__name__)


class GPyModel(BaseModel):

    def __init__(self, kernel, noise_variance=None, num_restarts=10, *args, **kwargs):
        """
        Interface to the GPy library. The GP hyperparameter are
        obtained by optimizing the marginal loglikelihood.

        Parameters
        ----------
        kernel : gpy kernel object
            Specifies the kernel that is used for all Gaussian Process
        prior : prior object
            Defines a prior for the hyperparameters of the GP. Make sure that
            it implements the Prior interface. During MCMC sampling the
            lnlikelihood is multiplied with the prior.
        noise_variance: float
            Noise term that is added to the diagonal of the covariance matrix
            for the cholesky decomposition.
        num_restarts: int
            Determines how often the optimization procedure for maximizing
            the marginal lln is restarted from different random points.
        """

        self.kernel = kernel
        self.noise_variance = noise_variance
        self.num_restarts = num_restarts
        self.X_star = None
        self.f_star = None
        self.m = None
        self.start_point = None

    def train(self, X, Y, do_optimize=True, **kwargs):
        """
        Computes the cholesky decomposition of the covariance of X and
        estimates the GP hyperparameter by optiminzing the marginal
        loglikelihood. The piror mean of the GP is set to the
        empirical mean of X.

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
        self.Y = Y
        if X.size == 0 or Y.size == 0:
            return

        kern = deepcopy(self.kernel)
        self.m = GPy.models.GPRegression(self.X, self.Y, kern)

        if self.noise_variance is not None:
            logger.warning("Do not optimize noise use fix value of %f" % (self.noise_variance))
            self.m.likelihood.variance.fix(self.noise_variance)
        else:
            # Add an exponential prior for the noise in order to prevent
            # that the GP explains everything with noise
            self.m.likelihood.unconstrain()
            self.m.likelihood.variance.set_prior(GPy.priors.Exponential(1))
            self.m.likelihood.variance.constrain_positive()

        if do_optimize:
            # Start from previous hyperparameter configuration
            self.m.optimize(start=self.start_point)

            logger.debug("HYPERS: " + str(self.m.param_array))
            self.start_point = self.m.param_array

        self.hypers = self.m.param_array

        self.observation_means = self.predict(self.X)[0]
        index_min = np.argmin(self.observation_means)
        self.X_star = self.X[index_min]
        self.f_star = self.observation_means[index_min]

    def predict_variance(self, X1, X2):
        r"""
        Predicts the variance between two test points X1, X2 by
           math: \sigma(X_1, X_2) = k_{X_1,X_2} - k_{X_1,X} * (K_{X,X}
                + \sigma^2*\mathds{I})^-1 * k_{X,X_2})

        Parameters
        ----------
        X1: np.ndarray (N, D)
            First test point
        X2: np.ndarray (1, D)
            Second test point
        Returns
        ----------
        np.array(N,1)
            predictive variance

        """
        kern = self.m.kern
        KbX = kern.K(X2, self.m.X).T
        Kx = kern.K(X1, self.m.X).T
        WiKx = np.dot(self.m.posterior.woodbury_inv, Kx)
        Kbx = kern.K(X2, X1)
        var = Kbx - np.dot(KbX.T, WiKx)

        return var.T

    def predict(self, X, full_cov=False, **kwargs):
        """
        Returns the predictive mean and variance of the objective function at
        the specified test point.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input test points
        full_cov: bool
            If set to true the full covariance between
            the test point and all observed points is returned

        Returns
        ----------
        np.array(N,1)
            predictive mean
        np.array(N,1)
            predictive variance

        """

        if self.m is None:
            logger.error("ERROR: Model has to be trained first.")
            return None

        mean, var = self.m.predict(X, full_cov=full_cov)

        if not full_cov:
            # GPy sometimes returns negative variance if the noise level is
            # too low clip them to be in the interval between the
            # smallest positive number and inf
            return mean, np.clip(var, np.finfo(var.dtype).eps, np.inf)

        else:
            # If we compute the full covariance matrix only clip the values on
            # the diagonal
            var[np.diag_indices(var.shape[0])] = np.clip(
                                            var[np.diag_indices(var.shape[0])],
                                            np.finfo(var.dtype).eps, np.inf)
            var[np.where((var < np.finfo(var.dtype).eps) & (var > -np.finfo(var.dtype).eps))] = 0

            return mean, var

    def predictive_gradients(self, Xnew):
        dmdx, dvdx = self.m.predictive_gradients(Xnew)
        return dmdx[:, 0, :], dvdx

    def get_noise(self):
        return self.m.likelihood.variance[0]

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

        return self.m.posterior_samples_f(X_test, n_funcs).T
