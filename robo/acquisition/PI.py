from scipy.stats import norm
import numpy as np

from robo.acquisition.base import AcquisitionFunction


class PI(AcquisitionFunction):
    r"""
    Probability of Improvement solves the following equation
    :math:`PI(X) := \mathbb{P}\left( f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) > \xi\right)`, where
    :math:`f(X^+)` is the best input found so far.

    :param model:  A Model that implements at least

                 - predict(X)

               If you want to calculate derivatives than it should also support

                 - predictive_gradients(X)
    :param X_lower: Lower bounds for the search, its shape should be 1xn (n = dimension of input space)
    :type X_lower: np.ndarray (1,n)
    :param X_upper: Upper bounds for the search, its shape should be 1xn (n = dimension of input space)
    :type X_upper: np.ndarray (1,n)
    :param par: A parameter meant to control the balance between exploration and exploitation of the acquisition
                function. Empirical testing determines 0.01 to be a good value in most cases.
    """
    long_name = "Probability of Improvement"

    def __init__(self, model, X_lower, X_upper, compute_incumbent, par=0.1, **kwargs):

        self.par = par
        self.compute_incumbent = compute_incumbent
        super(PI, self).__init__(model, X_lower, X_upper)

    def compute(self, X, derivative=False, **kwargs):
        """
        A call to the object returns the PI and derivative values.

        :param x: The point at which the function is to be evaluated.
        :type x: np.ndarray (1,n)
        :param compute_incumbent: Recommendation strategy that computes the incumbent
        :type function:
        :param derivative: This controls whether the derivative is to be returned.
        :type derivative: Boolean
        :return: The value of PI and its derivative at x.
        """
        if X.shape[0] > 1:
            print "PI is only for single x inputs"
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
