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
        raise NotImplementedError
    def sample_from_measure(self, x_prev, xmin, xmax, n_representers, BestGuesses, acquisition_fn):

        # If there are no prior observations, do uniform sampling
        if (x_prev.size == 0):
            dim = xmax.size
            zb = np.add(np.multiply((xmax - xmin), np.random.uniform(size=(n_representers, dim))), xmin)
            mb = np.dot(-np.log(np.prod(xmax - xmin)), np.ones((n_representers, 1)))
            return zb, mb
        # There are prior observations, i.e. it's not the first ES iteration
        dim = x_prev.size





    
class EI(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        raise NotImplementedError
    def model_changed(self):
        pass
