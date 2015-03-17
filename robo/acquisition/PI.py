from scipy.stats import norm
import numpy as np
from robo import BayesianOptimizationError 
from robo.acquisition.base import AcquisitionFunction 
class PI(AcquisitionFunction):
    
    long_name = "Probability of Improvement" 
    def __init__(self, model, X_lower, X_upper, par=0.1, **kwargs):
        self.model = model
        self.par = par
        self.X_lower = X_lower
        self.X_upper = X_upper

    def __call__(self, X, Z=None, derivative=False, **kwargs):
        if X.shape[0] > 1 :
            raise BayesianOptimizationError(BayesianOptimizationError.SINGLE_INPUT_ONLY, "PI is only for single x inputs")
        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            if derivative:
                f = 0
                df = np.zeros((1, X.shape[1]))
                return np.array([[f]]), np.array([df])
            else:
                return np.array([[0]])

        
        dim = X.shape[1]
        m, v = self.model.predict(X, Z)
        eta = self.model.getCurrentBest()
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
        if derivative:
            if len(df.shape) == 3:
                return_df = df
            else:
                return_df = np.array([df])
                
            return return_f, return_df
        else:
            return return_f
