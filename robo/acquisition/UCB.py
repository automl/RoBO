import numpy as np

from robo.acquisition.base import AcquisitionFunction
from robo import BayesianOptimizationError

class UCB(AcquisitionFunction):
    r"""
    The upper confidence bound is in this case a lower confidence bound.   
    
    .. math:: 
       
       UCB(X) := \mu(X) + \kappa\sigma(X)
    
    :param model: A model that implements at least 
        
                 - predict(X) 
    :param X_lower: Lower bounds for the search, its shape should be 1xD (D = dimension of search space)
    :type X_lower: np.ndarray (1,D)
    :param X_upper: Upper bounds for the search, its shape should be 1xD (D = dimension of search space)
    :type X_upper: np.ndarray (1,D)
    :param par: A parameter (:math:`\kappa`) meant to control the balance between exploration and exploitation of the acquisition
                function. 
    """
    long_name = "Upper Confidence Bound" 
    def __init__(self, model, X_lower, X_upper, par=1.0, **kwargs):
        self.model = model
        self.par = par
        self.X_lower = np.array(X_lower)
        self.X_upper = np.array(X_upper)
    def __call__(self, X, derivative=False,**kwargs):
        
        if derivative:
            raise BayesianOptimizationError(BayesianOptimizationError.NO_DERIVATIVE,
                                            "UCB  does not support derivative calculation until now")
        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            return np.array([[- np.finfo(np.float).max]])
        mean, var = self.model.predict(X)
        #minimize in f so maximize negative lower bound
        return -(mean - self.par * np.sqrt(var))
    def update(self, model):
        self.model = model