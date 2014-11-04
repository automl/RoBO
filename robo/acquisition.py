from scipy.stats import norm
import numpy as np
        
class pi_fkt(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        # print "*"*30,X
        mean, var = self.model.predict(X, Z)
        #print mean, var
        Y_star = self.model.getCurrentBest()
        u = 1 - norm.cdf((mean - Y_star) / var)
        #print "X", X, "MEAN", mean, "VAR", var,
        #print  "U", u
        return u


class ucb_fkt(object):
    def __init__(self, model):
        self.model = model
        
    def __call__(self, X, Z=None, **kwargs):
        mean, var = self.model.predict(X, Z)
        return -mean + var
    


