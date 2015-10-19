# encoding=utf8
import logging
from scipy.stats import norm
import numpy as np

from robo.acquisition.base import AcquisitionFunction

logger = logging.getLogger(__name__)

class EI(AcquisitionFunction):
    r"""
        Expected Improvement computes for a given x the acquisition value by
        :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi\right] \} ]`, with
        :math:`f(X^+)` as the incumbent.

        :param model: A model that implements at least

                 - predict(X)
                 - getCurrentBestX().

               If you want to calculate derivatives than it should also support

                 - predictive_gradients(X)
        :param X_lower: Lower bounds for the search, its shape should be 1xD (D = dimension of input space)
        :type X_lower: np.ndarray (1,D)
        :param X_upper: Upper bounds for the search, its shape should be 1xD (D = dimension of input space)
        :type X_upper: np.ndarray (1,D)
        :param compute_incumbent: A python function that takes as input a model and returns a np.array as incumbent
        :param par: Controls the balance between exploration and exploitation of the acquisition
                    function. Default is 0.01
    """

    long_name = "Expected Improvement"

    def __init__(self, model, X_lower, X_upper, compute_incumbent, par=0.01, **kwargs):
        #self.model = model
        self.par = par
        #self.X_lower = X_lower
        #self.X_upper = X_upper
        self.compute_incumbent = compute_incumbent
        super(EI, self).__init__(model, X_lower, X_upper)

        logger.debug("Test")

    def compute(self, X, derivative=False, **kwargs):
        """
        Computes the EI value and its derivatives.

        :param X: The point at which the function is to be evaluated.
        :type X: np.ndarray (1,D)
        :param derivative: This controls whether the derivative is to be returned.
        :type derivative: Boolean
        :return: The value of EI and optionally its derivative at X.
        :rtype: np.ndarray(1, 1) or (np.ndarray(1, 1), np.ndarray(1, D))
        """

        if X.shape[0] > 1:
            raise ValueError("EI is only for single test points")

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            if derivative:
                f = 0
                df = np.zeros((1, X.shape[1]))
                return np.array([[f]]), np.array([df])
            else:
                return np.array([[0]])

        m, v = self.model.predict(X, full_cov=True)
        incumbent, _ = self.compute_incumbent(self.model)
        eta, _ = self.model.predict(np.array([incumbent]))

        s = np.sqrt(v)
        z = (eta - m - self.par) / s
        f = (eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)
        if derivative:
            dmdx, ds2dx = self.model.predictive_gradients(X)
            dmdx = dmdx[0]
            ds2dx = ds2dx[0][:, None]
            dsdx = ds2dx / (2 * s)
            df = (-dmdx * norm.cdf(z) + (dsdx * norm.pdf(z))).T
        if (f < 0).any():
            f[np.where(f < 0)] = 0.0
            if derivative:
                df[np.where(f < 0), :] = np.zeros_like(X)
        if (f < 0).any():
            raise Exception
        if len(f.shape) == 1:
            f = np.array([f])
        if derivative:
            if len(df.shape) == 3:
                return_df = df
            else:
                return_df = np.array([df])
            return f, return_df
        else:
            return f
