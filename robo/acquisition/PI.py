from scipy.stats import norm
import numpy as np
class PI(object):
    def __init__(self, model, X_lower, X_upper, par=0.1, **kwargs):
        self.model = model
        self.par = par
        self.X_lower = X_lower
        self.X_upper = X_upper

    def __call__(self, X, Z=None, derivative=False, **kwargs):
        # TODO: add a parameter to condition the derivative being returned
        if (X < self.X_lower).any() or (X > self.X_upper).any():
            if derivative:
                u = 0
                du = np.zeros((X.shape[1],1))
                return u, du
            else:
                return 0

        alpha = np.linalg.solve(self.model.cK, np.linalg.solve(self.model.cK.transpose(), self.model.Y))
        dim = X.shape[1]
        m, v = self.model.predict(X, Z)
        eta = self.model.getCurrentBest()
        s = np.sqrt(v)
        z = (eta - m) / s - self.par
        f = norm.cdf(z)
        if derivative:
            dmdx, ds2dx = self.model.m.predictive_gradients(X)
            dsdx = ds2dx / (2*s)
            df = -(- norm.pdf(z) / s) * (dmdx + dsdx * z)
            return f, df
        else:
            return f

    def update(self, model):
        self.model = model