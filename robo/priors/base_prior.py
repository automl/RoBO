'''
Created on Oct 14, 2015

@author: Aaron Klein
'''

class BasePrior(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        pass

    def lnprob(self, x):
        '''
        Constructor
        '''
        return 0
    
    def sample(self, n_samples):
        return np.random.rand(n_samples)