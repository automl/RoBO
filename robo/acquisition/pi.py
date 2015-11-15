import logging
from scipy.stats import norm
import numpy as np

from robo.acquisition.base import AcquisitionFunction

logger = logging.getLogger(__name__)


class PI(AcquisitionFunction):

    def __init__(self, model, X_lower, X_upper, compute_incumbent, par=0.1, **kwargs):
        r"""
        Probability of Improvement solves the following equation
        :math:`PI(X) := \mathbb{P}\left( f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) > \xi\right)`, where
        :math:`f(X^+)` is the best input found so far.

        Parameters
        ----------
        model: Model object
            A model that implements at least
                 - predict(X)
                 - getCurrentBestX().
            If you want to calculate derivatives than it should also support
                 - predictive_gradients(X)

        X_lower: np.ndarray (D)
            Lower bounds of the input space
        X_upper: np.ndarray (D)
            Upper bounds of the input space
        compute_incumbent: func
            A python function that takes as input a model and returns
            a np.array as incumbent
        par: float
            Controls the balance between exploration
            and exploitation of the acquisition function. Default is 0.01
        """
        self.par = par
        self.compute_incumbent = compute_incumbent
        super(PI, self).__init__(model, X_lower, X_upper)

    def compute(self, X, derivative=False, **kwargs):
        """
        Computes the PI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(1, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            If is set to true also the derivative of the acquisition
            function at X is returned

        Returns
        -------
        np.ndarray(1,1)
            Probability of Improvement of X
        np.ndarray(1,D)
            Derivative of Probability of Improvement at X
            (only if derivative=True)
        """
        if X.shape[0] > 1:
            logger.error("PI is only for single x inputs")
            return
        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            if derivative:
                f = 0
                df = np.zeros((1, X.shape[1]))
                return np.array([[f]]), np.array([df])
            else:
                return np.array([[0]])

        m, v = self.model.predict(X)
        incumbent, _ = self.compute_incumbent(self.model)
        eta, _ = self.model.predict(np.array([incumbent]))
        s = np.sqrt(v)
        z = (eta - m - self.par) / s
        f = norm.cdf(z)
        if derivative:
            dmdx, ds2dx = self.model.predictive_gradients(X)
            dmdx = dmdx[0]
            ds2dx = ds2dx[0][:, None]
            dsdx = ds2dx / (2 * s)
            df = (-(-norm.pdf(z) / s) * (dmdx + dsdx * z)).T

        if len(f.shape) == 1:
            return_f = np.array([f])
        else:
            return_f = f
        if derivative:
            if len(df.shape) == 3:
                return_df = df
            else:
                return_df = np.array([df])

            return return_f, return_df
        else:
            return return_f
