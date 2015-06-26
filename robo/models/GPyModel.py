import sys
import StringIO
import numpy as np
import GPy
from robo.models.base_model import BaseModel


class GPyModel(BaseModel):
    """
     Wraps the standard Gaussian process for regression from the GPy library
    """
    def __init__(self, kernel, noise_variance=None, optimize=True, num_restarts=100, *args, **kwargs):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.optimize = optimize
        self.num_restarts = num_restarts
        self.X_star = None
        self.f_star = None
        self.m = None

    """def __getstate__(self):
        return dict(kernel = self.kernel,
                    noise_variance = self.noise_variance,
                    optimize = self.optimize,
                    num_restarts = self.num_restarts,
                    X = self.X,
                    Y = self.Y,
                    m_gaussian_noise = self.m['.*Gaussian_noise.variance'],
                    )"""

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        if X.size == 0 or Y.size == 0:
            return
        self.m = GPy.models.GPRegression(self.X, self.Y, self.kernel)
        self.m.constrain_positive('')

        self.likelihood = self.m.likelihood
        self.m[".*variance"].constrain_positive()
        if self.noise_variance is not None:
            # self.m['.*Gaussian_noise.variance'].unconstrain()
            # self.m.constrain_fixed('noise',self.noise_variance)
            print "constraining noise variance to ", self.noise_variance
            self.m['.*Gaussian_noise.variance'] = self.noise_variance
            self.m['.*Gaussian_noise.variance'].fix()

        if self.optimize:
            stdout = sys.stdout
            sys.stdout = StringIO.StringIO()
            self.m.optimize_restarts(num_restarts=self.num_restarts, robust=True)
            sys.stdout = stdout

        self.observation_means = self.predict(self.X)[0]
        index_min = np.argmin(self.observation_means)
        self.X_star = self.X[index_min]
        self.f_star = self.observation_means[index_min]

    def predict_variance(self, X1, X2):
        kern = self.m.kern
        KbX = kern.K(X2, self.m.X).T
        Kx = kern.K(X1, self.m.X).T
        WiKx = np.dot(self.m.posterior.woodbury_inv, Kx)
        Kbx = kern.K(X2, X1)
        var = Kbx - np.dot(KbX.T, WiKx)
        return var

    def predict(self, X, full_cov=False):
        if self.m == None:
            print "ERROR: Model has to be trained first."
            return None

        mean, var = self.m.predict(X, full_cov=full_cov)

        if not full_cov:
            return mean[:, 0], np.clip(var[:, 0], np.finfo(var.dtype).eps, np.inf)

        else:
            var[np.diag_indices(var.shape[0])] = np.clip(var[np.diag_indices(var.shape[0])], np.finfo(var.dtype).eps, np.inf)
            var[np.where((var < np.finfo(var.dtype).eps) & (var > -np.finfo(var.dtype).eps))] = 0
            return mean[:, 0], var

    def predictive_gradients(self, Xnew, X=None):
        if X is None:
            return self.m.predictive_gradients(Xnew)

    def sample(self, X, size=10):
        """
        samples from the GP at values X size times.
        """
        return self.m.posterior_samples_f(X, size)

    def visualize(self, ax, plot_min, plot_max):
        self.m.plot(ax=ax, plot_limits=[plot_min, plot_max])
