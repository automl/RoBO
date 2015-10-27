'''
Created on Oct 14, 2015

@author: Aaron Klein
'''

class BasePrior(object):

    def __init__(self):
        pass

    def lnprob(self, theta):
        pass
    
    def sample_from_prior(self, n_samples):
        return np.random.rand(n_samples)
    
    def gradient(self, theta):
        pass