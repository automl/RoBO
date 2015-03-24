# encoding=utf8



from scipy.stats import norm
import scipy
import numpy as np
from robo import BayesianOptimizationError 
from robo.acquisition.base import AcquisitionFunction 

class EI(AcquisitionFunction):
    r"""
        Expected Improvement solves the following equation
        :math:`\mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi\right] \} ]`, where 
        :math:`f(X^+)` is the best input found so far. 
        
        :param model: A Model that implements at least predict(X) and getCurrentBestX(). If you want to calculate derivatives than it should also support
                      predictive_gradients(X)  
        :param X_lower: Lower bounds for the search, its shape should be 1xn (n = dimension of input space)
        :type X_lower: np.ndarray (1,n)
        :param X_upper: Upper bounds for the search, its shape should be 1xn (n = dimension of input space)
        :type X_upper: np.ndarray (1,n)
        :param par: A parameter (:math:`\xi`) meant to control the balance between exploration and exploitation of the acquisition
                    function. Empirical testing determines 0.01 to be a good value in most cases. 
    """
    long_name = "Expected Improvement" 

    def __init__(self, model, X_lower, X_upper, par = 0.01, **kwargs):
        self.model = model
        self.par = par
        self.X_lower = X_lower
        self.X_upper = X_upper

    def __call__(self, X, derivative=False, **kwargs):
        """
        A call to the object returns the EI and derivative values.

        :param X: The point at which the function is to be evaluated.
        :type X: np.ndarray (1,n)
        :param derivative: This controls whether the derivative is to be returned.
        :type derivative: Boolean
        :return: The value of EI and optionally its derivative at X.
        :rtype: np.ndarray(N, 1) or (np.ndarray(N, 1), np.ndarray(N, D))  
        """
        if X.shape[0] > 1 :
            raise BayesianOptimizationError(BayesianOptimizationError.SINGLE_INPUT_ONLY, "EI is only for single X inputs")
        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            if derivative:
                f = 0
                df = np.zeros((1, X.shape[1]))
                return np.array([[f]]), np.array([df])
            else:
                return np.array([[0]])

        dim = X.shape[-1]
        m, v = self.model.predict(x)
        eta, _ = self.model.predict(np.array([self.model.getCurrentBestX()]))
        
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
                df[np.where(f < 0), :] = np.zeros_like(x)
        if (f < 0).any():
            raise Exception
        if len(f.shape) == 1:
            return_f = np.array([f])
        if derivative:
            if len(df.shape) == 3:
                return_df = df
            else:
                return_df = np.array([df])
                
            return return_f, return_df
        else:
            return return_f
        
