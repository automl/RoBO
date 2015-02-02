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
        mean, var = self.model.predict(X, Z)
        Y_star = self.model.getCurrentBest()
        u = norm.cdf((Y_star - mean - self.par ) / var)
        if derivative:
            # Derivative values:
            # Derivative of kernel values:
            dkxX = self.model.kernel.gradients_X(np.array([np.ones(len(self.model.X))]), self.model.X, X)
            dkxx = self.model.kernel.gradients_X(np.array([np.ones(len(self.model.X))]), self.model.X)
            # dmdx = derivative of the gaussian process mean function
            dmdx = np.dot(dkxX.transpose(), alpha)
            # dsdx = derivative of the gaussian process covariance function
            dsdx = np.zeros((dim, 1))
            for i in range(0, dim):
                dsdx[i] = np.dot(0.5 / var, dkxx[0,dim-1] - 2 * np.dot(dkxX[:,dim-1].transpose(),
                                                                       np.linalg.solve(self.model.cK,
                                                                                       np.linalg.solve(self.model.cK.transpose(),
                                                                                                       self.model.K[0,None].transpose()))))
            # (-phi/s) * (dmdx + dsdx * z)
            z = (Y_star - mean) / var
            du = (- norm.pdf(z) / var) * (dmdx + dsdx * z)
            return u, du
        else:
            return u

    def update(self, model):
        self.model = model