"""
this module contains different model classes that can be used by the Bayesian
Optimization

thios model interface is taken from  the `bitbucket wiki <http://https://bitbucket.org/aadfreiburg/robo/wiki/Home>`_

.. class:: Model
    

    .. method:: Train (X,Z,Y)
     
       :params Z:  are the instance features
    
    .. method:: marginalize features
    
    .. method:: conditionalize features
    
    .. method:: predict (X,Z)
     
       :returns mean, variance:
       
    .. method:: update(X,Z,Y)
    
    .. method:: downdate()
    
    .. method:: load() 
    
    .. method:: save()
    
    other wishes:
    
        * Training should support approximations
        * Training should support an interface for optimizing the hyperparameters of the model
        * interface to compute the information gain
        * interface to draw sample from the model
        * validate()

"""

import sys
import StringIO
import numpy as np
import GPy as GPy

class GPyModel(object):
    """
    GPyModel is just a wrapper around the GPy Lib
    """
    def __init__(self, kernel, noise_variance = None, optimize=True, *args, **kwargs):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.optimize = optimize
        
        
    def train(self, X, Y,  Z=None):
        self.X = X
        self.Y = Y
        if X.size == 0 or Y.size == 0:
            return
        self.Z = Z
        self.m = GPy.models.GPRegression(self.X, self.Y, self.kernel)
        self.likelihood = self.m.likelihood
        #old gpy 
        if self.noise_variance is not None:
            try:
                self.m.constrain_fixed('.*noise', self.noise_variance)
            #gpy version >=0.6
            except:
                self.m['.*Gaussian_noise.variance'] = self.noise_variance
                self.m['.*Gaussian_noise.variance'].unconstrain()
                self.m['.*Gaussian_noise.variance'].fix()
        if self.optimize:
            self.m.optimize_restarts(num_restarts = 10, robust=True)

        index_min = np.argmin(self.Y)
        self.X_star = self.X[index_min]
        self.Y_star = self.Y[index_min]

    def update(self, X, Y, Z=None):
        X = np.append(self.X, X, axis=0)
        Y = np.append(self.Y, Y, axis=0)
        if self.Z != None:
            Z = np.append(self.Z, [Z], axis=0)
        self.train(X, Y, Z)

    def predict(self, X, Z=None, full_cov=False):
        #old gpy version 
        try:
            mean, var, _025pm, _975pm = self.m.predict(X, full_cov=full_cov)
            if not full_cov:
                return mean[:,0], var[:,0]
            else:
                return mean[:,0], var
        #gpy version >=0.6
        except (ValueError, AssertionError):
            mean, var = self.m.predict(X, full_cov=full_cov)
            if not full_cov:
                return mean[:,0], var[:,0]
            else:
                return mean[:,0], var
            
    def load(self, filename):
        pass
    
    def save(self, filename):
        pass
    
    def visualize(self):
        pass
    
    def getCurrentBest(self):
        return self.Y_star
        