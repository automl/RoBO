import numpy as np
import GPy as GPy

class GPyModel(object):
    """
    GPyModel is just a wrapper around the GPy Lib
    """
    def __init__(self, kernel, *args, **kwargs):
        self.kernel = GPy.kern.rbf(input_dim=1, variance=270**2, lengthscale=0.2)
    
    def train(self, X, Y,  Z=None):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.m = GPy.models.GPRegression(self.X, self.Y, self.kernel)
        self.m.unconstrain('')
        self.m.constrain_positive('.*rbf_variance')
        self.m.constrain_bounded('.*lengthscale', 1., 10.)
        self.m.constrain_fixed('.*noise', 0.0025)
        self.m.optimize()
        self.X_star = self.X[np.argmax(self.Y)]
        self.Y_star = np.max(self.Y)
        
    def update(self, X, Y, Z=None):
        self.X = np.append(self.X, [X], axis=0)
        self.Y = np.append(self.Y, [Y], axis=0)
        if self.Z != None:
            self.Z = Z
        self.m = GPy.models.GPRegression(self.X, self.Y, self.kernel)
        self.m.unconstrain('')
        self.m.constrain_fixed('.*noise', 0.0025)
        self.m.constrain_positive('.*rbf_variance')
        self.m.constrain_bounded('.*lengthscale', 1., 10.)
        self.m.optimize()
        if Y < self.Y_star:
            self.Y_star = Y
    
    def predict(self, X, Z=None):
        mean, var, _025pm, _975pm = self.m.predict(X)
        
        return mean, np.sqrt(var)
    
    def load(self, filename):
        pass
    
    def save(self, filename):
        pass
    
    def visualize(self):
        pass
    
    def getCurrentBest(self):
        return self.Y_star
        