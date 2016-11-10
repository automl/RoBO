import logging
import numpy as np

from robo.acquisition_functions.base_acquisition import BaseAcquisitionFunction


logger = logging.getLogger(__name__)


class LCB(BaseAcquisitionFunction):

    def __init__(self, model, par=1.0):
        r"""
        The lower confidence bound that computes for a
        test point the acquisition value by:

        .. math::

        LCB(X) := - (\mu(x) - \kappa\sigma(x))

        Note: We want to find the minimum of and thus minimize the lower confidence bound.
        But RoBO always maximizes the acquisition function

        Parameters
        ----------
        model: Model object
            A model that implements at least
                 - predict(X)
                 - getCurrentBestX().
            If you want to calculate derivatives than it should also support
                 - predictive_gradients(X)

        par: float
            Controls the balance between exploration
            and exploitation of the acquisition_functions function. Default is 1
        """
        self.par = par
        super(LCB, self).__init__(model)

    def compute(self, X, derivative=False, **kwargs):
        """
        Computes the LCB acquisition_functions value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input point where the acquisition_functions function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            If is set to true also the derivative of the acquisition_functions
            function at X is returned.

        Returns
        -------
        np.ndarray(N,)
            LCB value of X
        np.ndarray(N,D)
            Derivative of LCB at X (only if derivative=True)
        """
        mean, var = self.model.predict(X)

        # RoBO maximizes the acquisition function but we want to minimize the lower confidence bound
        acq = - (mean - self.par * np.sqrt(var))
        if derivative:
            dm, dv = self.model.predictive_gradients(X)
            grad = -(dm - self.par * dv / (2 * np.sqrt(var)))
            return acq, grad
        else:
            return acq
