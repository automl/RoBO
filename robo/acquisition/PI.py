from scipy.stats import norm
import numpy as np
from robo import BayesianOptimizationError 
from robo.acquisition.base import AcquisitionFunction 
class PI(AcquisitionFunction):
    def __init__(self, model, X_lower, X_upper, par=0.1, **kwargs):
        self.model = model
        self.par = par
        self.X_lower = X_lower
        self.X_upper = X_upper

    def __call__(self, X, Z=None, derivative=False, **kwargs):
        if X.shape[0] > 1 :
            raise BayesianOptimizationError(BayesianOptimizationError.SINGLE_INPUT_ONLY, "EI is only for single x inputs")
        if (X < self.X_lower).any() or (X > self.X_upper).any():
            if derivative:
                f = 0
                df = np.zeros((X.shape[1],1))
                return np.array([[f]]), np.array([[df]])
            else:
                return np.array([[0]])

        alpha = np.linalg.solve(self.model.cK, np.linalg.solve(self.model.cK.transpose(), self.model.Y))
        dim = X.shape[1]
        m, v = self.model.predict(X, Z)
        eta = self.model.getCurrentBest()
        s = np.sqrt(v)
        z = (eta - m - self.par) / s 
        f = norm.cdf(z)
        if derivative:
            dmdx, ds2dx = self.model.predictive_gradients(X)
            dsdx = ds2dx / (2*s)
            df = -(- norm.pdf(z) / s) * (dmdx + dsdx * z)
        
            return np.array([f]), np.array([df])
        else:
            return np.array([f])
