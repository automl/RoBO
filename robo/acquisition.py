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
    def model_changed(self):
        pass

class UCB(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        mean, var = self.model.predict(X, Z)
        return -mean + var
    def model_changed(self):
        pass

class Entropy(object):
    # This function calls PI, EI etc and samples them (using their values)
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        raise NotImplementedError
    def model_changed(self):
        pass
    def sample_from_measure(self, x_prev, xmin, xmax, n_representers, BestGuesses, acquisition_fn):
        if
        pass

    
class EI(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        raise NotImplementedError
    def model_changed(self):
        pass
