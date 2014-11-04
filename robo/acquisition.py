from scipy.stats import norm
import numpy as np
        
class PI(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        mean, var = self.model.predict(X, Z)
        Y_star = self.model.getCurrentBest()
        u = 1 - norm.cdf((mean - Y_star) / var)
        return u

class UCB(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        mean, var = self.model.predict(X, Z)
        return -mean + var

class Entropy(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        raise NotImplementedError
    
class EI(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        raise NotImplementedError
    
