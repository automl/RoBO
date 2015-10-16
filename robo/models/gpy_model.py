import sys
import StringIO
import numpy as np
import GPy
import logging

from scipy import spatial

from robo.models.base_model import BaseModel
from copy import deepcopy


class GPyModel(BaseModel):
    """
     Wraps the standard Gaussian process for regression from the GPy library
    """
    def __init__(self, kernel, noise_variance=None, num_restarts=10, *args, **kwargs):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.num_restarts = num_restarts
        self.X_star = None
        self.f_star = None
        self.m = None
        self.start_point = None

    def train(self, X, Y, do_optimize=True):
        self.X = X
        self.Y = Y
        if X.size == 0 or Y.size == 0:
            return
        
        kern = deepcopy(self.kernel)
        self.m = GPy.models.GPRegression(self.X, self.Y, kern)
        
        if self.noise_variance is not None:
            print "Do not optimize noise use fix value of %f" % (self.noise_variance)
            self.m.likelihood.variance.fix(self.noise_variance)
        else:
            # Add an exponential prior for the noise in order to prevent that the GP explains everything with noise
             self.m.likelihood.unconstrain()
             self.m.likelihood.variance.set_prior(GPy.priors.Exponential(1))
             self.m.likelihood.variance.constrain_positive()
        
        if do_optimize:
            # Start from previous hyperparameters
            self.m.optimize(start=self.start_point)
            # Start from random
            #self.m.optimize_restarts(num_restarts=self.num_restarts)
            logging.info("HYPERS: " + str(self.m.param_array))
            self.start_point = self.m.param_array

        self.hypers = self.m.param_array

        self.observation_means = self.predict(self.X)[0]
        index_min = np.argmin(self.observation_means)
        self.X_star = self.X[index_min]
        self.f_star = self.observation_means[index_min]

    def predict_variance(self, X1, X2):
        """
            Predict the variance between two test points X1, X2 by Sigma(X1, X2) = k_X1,X2 - k_X1,X * (K_X,X + simga^2*I)^-1 * k_X,X2)
        """
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
            # GPy sometimes returns negative variance if the noise level is too low, clip them to be in the interval between the smallest positive number and inf
            #if np.any(var < 0):
            #    logging.error("Variance is negative (%s)" % var)
            return mean[:, 0], np.clip(var[:, 0], np.finfo(var.dtype).eps, np.inf)

        else:
            # If we compute the full covariance matrix only clip the values on the diagonal
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
