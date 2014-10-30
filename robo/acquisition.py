from scipy.stats import norm
import numpy as np

def pi_fkt(model):
    def acq_fkt(X, Z=None, **kwargs):
        mean, var = model.predict(X, Z)
        Y_star = model.getCurrentBest()
        return 1-norm.cdf((mean-Y_star)/var)
    return acq_fkt
    
def ucb_fkt(model):
    def acq_fkt(X, Z=None, **kwargs):
        mean, var = model.predict(X, Z)
        return  -mean + var
    return acq_fkt