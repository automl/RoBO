import logging
import numpy as np

from robo.acquisition.base import AcquisitionFunction


logger = logging.getLogger(__name__)


class UCB(AcquisitionFunction):

    def __init__(self, model, X_lower, X_upper, par=1.0, **kwargs):
        r"""
        The upper confidence bound is in this case a lower confidence bound.

        .. math::

        UCB(X) := \mu(X) + \kappa\sigma(X)

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
        par: float
            Controls the balance between exploration
            and exploitation of the acquisition function. Default is 0.01
        """
        self.par = par
        super(UCB, self).__init__(model, X_lower, X_upper)

    def compute(self, X, derivative=False, **kwargs):

        """
        Computes the UCB value and its derivatives.

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
            UCB value of X
        np.ndarray(1,D)
            Derivative of UCB at X (only if derivative=True)
        """

        if derivative:
            logger.error("UCB  does not support derivative calculation until now")
            return
        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            return np.array([[- np.finfo(np.float).max]])
        mean, var = self.model.predict(X)
        # minimize in f so maximize negative lower bound
        return -(mean - self.par * np.sqrt(var))

    def update(self, model):
        self.model = model
