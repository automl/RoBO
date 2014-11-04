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
        self.m.constrain_fixed('.*noise', self.noise_variance)
        self.m.optimize()
        self.X_star = self.X[np.argmax(self.Y)]
        self.Y_star = np.max(self.Y)
        
    def update(self, X, Y, Z=None):
        #print self.X, self.Y
        self.X = np.append(self.X, [X], axis=0)
        self.Y = np.append(self.Y, [Y], axis=0)
        if self.Z != None:
            self.Z = Z
        self.m = GPy.models.GPRegression(self.X, self.Y, self.kernel)#, likelihood=likelihood)
        self.m.constrain_fixed('.*noise', self.noise_variance)
        self.m.optimize()
        if Y < self.Y_star:
            self.Y_star = Y
    
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
        