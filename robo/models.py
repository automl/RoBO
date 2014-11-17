import sys
import StringIO
import numpy as np
import GPy as GPy

class GPyModel(object):
    """
    GPyModel is just a wrapper around the GPy Lib
    """
    def __init__(self, kernel, noise_variance = 0.002,*args, **kwargs):
        self.kernel = kernel
        self.noise_variance = noise_variance
    def train(self, X, Y,  Z=None):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.m = GPy.models.GPRegression(self.X, self.Y, self.kernel)#, likelihood=likelihood)
        #stdout = sys.stdout
        #sys.stdout = StringIO.StringIO()
        self.m.constrain_fixed('.*noise', self.noise_variance)
        self.m.optimize()
        index_min = np.argmin(self.Y)
        self.X_star = self.X[index_min]
        self.Y_star = self.Y[index_min]
        #sys.stdout = stdout
    def update(self, X, Y, Z=None):
        #print self.X, self.Y
        X = np.append(self.X, X, axis=0)
        Y = np.append(self.Y, Y, axis=0)
        if self.Z != None:
            Z = np.append(self.Z, [Z], axis=0)
        self.train(X, Y, Z)
        
    
    def predict(self, X, Z=None):
        #print "X", X
        mean, var, _025pm, _975pm = self.m.predict(X)
        return mean[:,0], var[:,0]
    
    def load(self, filename):
        pass
    
    def save(self, filename):
        pass
    
    def visualize(self):
        pass
    
    def getCurrentBest(self):
        return self.Y_star
        