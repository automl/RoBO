import numpy as np
import GPy

from robo.models.base_model import BaseModel
from robo.models.GPyModel import GPyModel


class GPyModelMCMC(BaseModel):

    def __init__(self, kernel, noise_variance=None, burnin=200, chain_length=100, n_hypers=10, *args, **kwargs):

        assert chain_length >= n_hypers

        self.kernel = kernel
        self.noise_variance = noise_variance
        self.X_star = None
        self.f_star = None
        self.models = None
        self.burnin = burnin
        self.chain_length = chain_length
        self.n_hypers = n_hypers

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        if X.size == 0 or Y.size == 0:
            return

        m = GPy.models.GPRegression(self.X, self.Y, self.kernel)
        hmc = GPy.inference.mcmc.hmc.HMC(m, stepsize=5e-2)
        s = hmc.sample(num_samples=(self.burnin + self.chain_length))
        self.hypers = s[range(self.burnin, self.burnin + self.chain_length, self.chain_length / self.n_hypers)]

        self.models = []
        for hyper in self.hypers:
            m = GPyModel(self.kernel, optimization=False)
            m.param_array = hyper
            m.train(self.X, self.Y)
            self.models.append(m)

    def predict(self, X, full_cov=False):
        if self.models == None:
            print "ERROR: The model needs to be trained first."
            return None

        mean = np.zeros([self.n_hypers, X.shape[0]])
        var = np.zeros([self.n_hypers, X.shape[0]])

        for i, model in enumerate(self.models):
                mean[i], var[i] = model.predict(X, full_cov)
        return mean, var
