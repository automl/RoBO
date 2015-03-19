import numpy as np

from robo.acquisition.base import AcquisitionFunction
from robo import BayesianOptimizationError

class UCB(AcquisitionFunction):
    long_name = "Upper Confidence Bound" 
    def __init__(self, model, X_lower, X_upper, par=1.0, **kwargs):
        self.model = model
        self.par = par
        self.X_lower = np.array(X_lower)
        self.X_upper = np.array(X_upper)
    def __call__(self, X, Z=None, derivative=False,**kwargs):
        
        if derivative:
            raise BayesianOptimizationError(BayesianOptimizationError.NO_DERIVATIVE,
                                            "UCB  does not support derivative calculation until now")
        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            return np.array([[- np.finfo(np.float).max]])
        mean, var = self.model.predict(X, Z)
        #minimize in f so maximize negative lower bound
        return -(mean - self.par * np.sqrt(var))
    def update(self, model):
        self.model = model