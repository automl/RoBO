
import scipy
import numpy as np

class EntropyMC(object):
    def __init__(self, model, X_lower, X_upper):
        self.model = model
        self.X_lower = X_lower
        self.X_upper = X_upper
    
    def __call__(self, X, Z=None, **kwargs):
        return self.dh_fun(X)

    def update(self, model):
        self.model = model
        #TODO  create distribution over pmin, 
        raise NotImplementedError()
    
    def dh_fun(self, x):
        #TODO
        raise NotImplementedError()