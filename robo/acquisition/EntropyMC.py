import sys
from scipy.stats import norm
import scipy
import numpy as np
from robo.loss_functions import logLoss
from robo import BayesianOptimizationError
from robo.sampling import sample_from_measure, montecarlo_sampler
from robo.acquisition.EI import EI
sq2 = np.sqrt(2)
l2p = np.log(2) + np.log(np.pi)
eps = np.finfo(np.float32).eps

class EntropyMC(object):
    def __init__(self, model, X_lower, X_upper, Nb = 100, sampling_acquisition = None, sampling_acquisition_kw = {"par":2.4}, T=200, loss_function=None, **kwargs):
        self.model = model
        self.Nb = Nb
        self.X_lower = np.array(X_lower)
        self.X_upper = np.array(X_upper)
        self.BestGuesses = np.zeros((0, X_lower.shape[0]))
        if sampling_acquisition is None:
            sampling_acquisition = EI
        self.sampling_acquisition = sampling_acquisition(model, self.X_lower, self.X_upper, **sampling_acquisition_kw)
        if loss_function is None:
            loss_function = logLoss
        self.loss_function = loss_function
        self.T = T
    
    def __call__(self, X, Z=None, **kwargs):
        return self.dh_fun(X, self.zb, self.lmb, self.pmin)

    def update(self, model):
        self.model = model
        self.sampling_acquisition.update(model)
        self.zb, self.lmb = sample_from_measure(self.model, self.X_lower, self.X_upper, self.Nb, self.BestGuesses, self.sampling_acquisition)
        # lmb = log_proposal_values; representer_points = zb; hallucinated_values= pmin approximation

        mu, var = self.model.predict(np.array(self.zb), full_cov=True)
        self.pmin = montecarlo_sampler(model, self.X_lower, self.X_upper, zb=self.zb)[1].T

        # return pmin[1]

    
    def dh_fun(self, x, zb, lmb, pmin):
        # TODO: should this be shape[1] ?
        n = pmin.shape[0]
        kl_divergence = 0

        for i in range(0, n):
            entropy_pmin = -np.dot(pmin.T, np.log(pmin + 1e-50))
            log_proposal = np.dot(lmb.T, pmin)
            kl_divergence += (entropy_pmin - log_proposal) / n
        return kl_divergence