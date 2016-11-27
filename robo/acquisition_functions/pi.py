import logging
import numpy as np

from scipy.stats import norm

from robo.acquisition_functions.base_acquisition import BaseAcquisitionFunction

logger = logging.getLogger(__name__)


class PI(BaseAcquisitionFunction):

    def __init__(self, model, par=0.0):
        r"""
        Probability of Improvement solves the following equation
        :math:`PI(X) := \mathbb{P}\left( f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) > \xi\right)`, where
        :math:`f(X^+)` is the best input found so far.

        Parameters
        ----------
        model: Model object
            Current belief of your objective function
            If you want to calculate derivatives than the model should also support
                 - predictive_gradients(X_test)

        par: float
            Controls the balance between exploration
            and exploitation of the acquisition_functions function.
        """
        super(PI, self).__init__(model)

        self.par = par

    def compute(self, X_test, derivative=False):
        """
        Computes the PI value and its derivatives.

        Parameters
        ----------
        X_test: np.ndarray(1, D), The input point where the acquisition_functions function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            If is set to true also the derivative of the acquisition_functions
            function at X is returned

        Returns
        -------
        np.ndarray(1,1)
            Probability of Improvement of X_test
        np.ndarray(1,D)
            Derivative of Probability of Improvement at X_test
            (only if derivative=True)
        """

        m, v = self.model.predict(X_test)
        _, inc_val = self.model.get_incumbent()

        s = np.sqrt(v)
        z = (inc_val - m - self.par) / s
        f = norm.cdf(z)

        if derivative:
            dmdx, ds2dx = self.model.predictive_gradients(X_test)
            dmdx = dmdx[0]
            ds2dx = ds2dx[0][:, None]
            dsdx = ds2dx / (2 * s)
            df = ((-norm.pdf(z) / s) * (dmdx + dsdx * z)).T
            return f, df
        else:
            return f
