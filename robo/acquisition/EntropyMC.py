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
        return self.dh_fun(X, self.zb, self.lmb)

    def update(self, model):
        self.model = model
        self.sampling_acquisition.update(model)
        self.zb, self.lmb = sample_from_measure(self.model, self.X_lower, self.X_upper, self.Nb, self.BestGuesses, self.sampling_acquisition)
        # lmb = log_proposal_values; representer_points = zb; hallucinated_values= pmin approximation
        self.W = np.random.multivariate_normal(mean=np.zeros(self.Nb), cov=np.eye(self.Nb), size=self.T)
        self.F = np.random.randn(self.T,1)



    
    def dh_fun(self, x, zb, lmb):
        import copy
        if x.shape[0] > 1:
            raise BayesianOptimizationError(BayesianOptimizationError.SINGLE_INPUT_ONLY, "dHdx_local is only for single x inputs")
        # print pmin.shape
        n = x.shape[0]
        kl_divergence = 0

        # Simulate the value of f at the candidate point, x
        new_y = self.model.sample(x, size=1)

        # Construct new GP model with the simulated observation
        sim_model = copy.deepcopy(self.model)
        sim_model.update(x, new_y)

        mu, var = sim_model.predict(zb, full_cov=True)
        cVar = np.linalg.cholesky(var)

        model_samples = np.add(np.dot(cVar.T, self.W.T).T, mu)

        # Approximate pmin:
        mins = np.argmin(model_samples, axis=1)
        min_count = np.zeros(model_samples.shape).T
        min_count[mins, np.arange(0, self.T)] = 1
        pmin = np.sum(min_count, axis=1) * (1. / self.T)

        # Calculate the Kullback-Leibler divergence w.r.t. this pmin approximation and return
        entropy_pmin = -np.dot(pmin, np.log(pmin + 1e-50))
        log_proposal = np.dot(lmb.T, pmin)
        kl_divergence += (entropy_pmin - log_proposal) / n

        return kl_divergence[0]