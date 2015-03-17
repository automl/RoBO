# encoding=utf8
from scipy.stats import norm
import scipy
import numpy as np
from robo import BayesianOptimizationError 
from robo.acquisition.base import AcquisitionFunction 

class EI(AcquisitionFunction):

    """
    When calling this object it will return the expected improvement at a point x, as well as the derivative value
    of the EI function. It works only for single input points.
    """

    long_name = "Expected Improvement" 

    def __init__(self, model, X_lower, X_upper, par = 0.01,**kwargs):
        """

        :param model: A GPyModel contatining current data points.
        :param X_lower: Lower bounds for the search, its shape should be 1xn (n = dimension of search space)
        :param X_upper: Upper bounds for the search, its shape should be 1xn (n = dimension of search space)
        :param par: A parameter meant to control the balance between exploration and exploitation of the acquisition
                    function. Empirical testing determines 0.01 to be a good value in most cases.

        :return: The value of the EI function and its derivative at point x.
        """
        
        self.model = model
        self.par = par
        self.X_lower = X_lower
        self.X_upper = X_upper

    def __call__(self, x, Z=None, derivative=False, **kwargs):
        if x.shape[0] > 1 :
            raise BayesianOptimizationError(BayesianOptimizationError.SINGLE_INPUT_ONLY, "EI is only for single x inputs")
        if np.any(x < self.X_lower) or np.any(x > self.X_upper):
            if derivative:
                f = 0
                df = np.zeros((1, x.shape[1]))
                return np.array([[f]]), np.array([df])
            else:
                return np.array([[0]])

        dim = x.shape[-1]
        m, v = self.model.predict(x)
        eta, _ = self.model.predict(np.array([self.model.getCurrentBestX()]))
        
        s = np.sqrt(v)
        z = (eta - m - self.par) / s 
        f = (eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)
        if derivative:
            dmdx, ds2dx = self.model.predictive_gradients(x)
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
        
