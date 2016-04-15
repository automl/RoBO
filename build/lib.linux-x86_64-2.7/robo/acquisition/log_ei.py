import logging
from scipy.stats import norm
import numpy as np

from robo.acquisition.base import AcquisitionFunction
from robo.incumbent.best_observation import BestObservation

logger = logging.getLogger(__name__)


class LogEI(AcquisitionFunction):

    def __init__(self, model, X_lower, X_upper, par=0.01, **kwargs):

        r"""
        Computes for a given x the logarithm expected improvement as
        acquisition value.

        Parameters
        ----------
        model: Model object
            A model that implements at least
                 - predict(X)
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

        super(LogEI, self).__init__(model, X_lower, X_upper)

        self.par = par
        self.rec = BestObservation(self.model, self.X_lower, self.X_upper)

    def update(self, model):
        """
        This method will be called if the model is updated.

        Parameters
        ----------
        model : Model object
            Models the objective function.
        """

        super(LogEI, self).update(model)
        self.rec = BestObservation(self.model, self.X_lower, self.X_upper)

    def compute(self, X, derivative=False, **kwargs):
        """
        Computes the Log EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(1, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            If is set to true also the derivative of the acquisition
            function at X is returned
            Not implemented yet!

        Returns
        -------
        np.ndarray(1,1)
            Log Expected Improvement of X
        np.ndarray(1,D)
            Derivative of Log Expected Improvement at X
            (only if derivative=True)
        """
        if derivative:
            logger.error("LogEI does not support derivative \
                calculation until now")
            return

        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            return np.array([[- np.finfo(np.float).max]])
        m, v = self.model.predict(X)

        _, eta = self.rec.estimate_incumbent(None)

        f_min = eta - self.par

        s = np.sqrt(v)

        z = (f_min - m) / s

        log_ei = np.zeros((m.size, 1))
        for i in range(0, m.size):
            mu, sigma = m[i], s[i]

        #    par_s = self.par * sigma

            # Degenerate case 1: first term vanishes
            if np.any(abs(f_min - mu)) == 0:
                if sigma > 0:
                    log_ei[i] = np.log(sigma) + norm.logpdf(z[i])
                else:
                    log_ei[i] = -np.Infinity
            # Degenerate case 2: second term vanishes and first term
            # has a special form.
            elif sigma == 0:
                if mu < np.any(f_min):
                    log_ei[i] = np.log(f_min - mu)
                else:
                    log_ei[i] = -np.Infinity
            # Normal case
            else:
                b = np.log(sigma) + norm.logpdf(z[i])
                # log(y+z) is tricky, we distinguish two cases:
                if np.any(f_min > mu):
                    # When y>0, z>0, we define a=ln(y), b=ln(z).
                    # Then y+z = exp[ max(a,b) + ln(1 + exp(-|b-a|)) ],
                    # and thus log(y+z) = max(a,b) + ln(1 + exp(-|b-a|))
                    a = np.log(f_min - mu) + norm.logcdf(z[i])

                    log_ei[i] = max(a, b) + np.log(1 + np.exp(-abs(b - a)))
                else:
                    # When y<0, z>0, we define a=ln(-y), b=ln(z),
                    # and it has to be true that b >= a in
                    # order to satisfy y+z>=0.
                    # Then y+z = exp[ b + ln(exp(b-a) -1) ],
                    # and thus log(y+z) = a + ln(exp(b-a) -1)
                    a = np.log(mu - f_min) + norm.logcdf(z[i])
                    if a >= b:
                        # a>b can only happen due to numerical inaccuracies
                        # or approximation errors
                        log_ei[i] = -np.Infinity
                    else:
                        log_ei[i] = b + np.log(1 - np.exp(a - b))

        return log_ei
