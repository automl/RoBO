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
        self.mcmc_chain = hmc.sample(num_samples=(self.burnin + self.chain_length))
        self.samples = self.mcmc_chain[range(self.burnin, self.burnin + self.chain_length, self.chain_length / self.n_hypers)]

        self.models = []
        for sample in self.samples:
            kernel = self.kernel
            for i in range(len(sample) - 1):
                kernel.parameters[i][0] = sample[i]
            model = GPyModel(kernel, noise_variance=sample[-1], optimization=False)
            model.train(self.X, self.Y)
            self.models.append(m)

    def predict(self, X, full_cov=False):
        if self.models == None:
            print "ERROR: The model needs to be trained first."
            return None

        mean = np.zeros([self.n_hypers, X.shape[0]])
        var = np.zeros([self.n_hypers, X.shape[0]])

        for i, model in enumerate(self.models):
            m, v = model.predict(X, full_cov)
            mean[i, :] = m[:, 0]
            var[i, :] = v[:, 0]

        return mean, var

    def predict_variance(self, X1, X2):
        var = []
        for m in self.models:
            var.append(m.predict_variance(X1, X2))

        return np.array(var)
