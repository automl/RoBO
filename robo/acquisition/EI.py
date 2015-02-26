#encoding=utf8
from scipy.stats import norm
import scipy
import numpy as np
from robo import BayesianOptimizationError 
from robo.acquisition.base import AcquisitionFunction 

class EI(AcquisitionFunction):
    r'''
    The expected improvement:
    :model:
    
    .. math::
       \begin{array}{rl}
            EI(\mathbf{x}) = \left\{ \begin{array}{ll}
               \left(\mu(\mathbf{x}) - f (\mathbf{x}^+) -
               \xi\sigma(\mathbf{x})\right)\Phi(Z) +
               \sigma(\mathbf{x})\varphi(Z) & if \sigma(\mathbf{x}) > 0 \\
               0 & if \sigma(\mathbf{x}) = 0
            \end{array}\right.
       \end{array}
    '''
    def __init__(self, model, X_lower, X_upper, par = 0.01,**kwargs):
        self.model = model
        self.par = par
        self.X_lower = X_lower
        self.X_upper = X_upper

    def __call__(self, x, Z=None, derivative=False,  **kwargs):
        if x.shape[0] > 1 :
            raise BayesianOptimizationError(BayesianOptimizationError.SINGLE_INPUT_ONLY, "EI is only for single x inputs")
        if (x < self.X_lower).any() or (x > self.X_upper).any():
            if derivative:
                f = 0
                df = np.zeros((x.shape[1],1))
                return np.array([[f]]), np.array([[df]])
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
            dsdx = ds2dx / (2*s)
            df = -dmdx * norm.cdf(z) + dsdx * norm.pdf(z)
        if (f < 0).any():
            f[np.where(f < 0)] = 0.0
            if derivative:
                df[np.where(f < 0), :] = np.zeros_like(x)
        if (f < 0).any():
            raise Exception
        if derivative:
            return np.array([f]), np.array([df])
        else:
            return np.array([f])
        
