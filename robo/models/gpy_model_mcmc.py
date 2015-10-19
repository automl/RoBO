import logging
import numpy as np
import GPy
from copy import deepcopy

from robo.models.base_model import BaseModel
from robo.models.gpy_model import GPyModel

logger = logging.getLogger(__name__)

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
        self.hmc = None

    def train(self, X, Y, *args):
        self.X = X
        self.Y = Y
        if X.size == 0 or Y.size == 0:
            return

        m = GPy.models.GPRegression(self.X, self.Y, self.kernel)
        # Add exponential prior for the noise
        m.likelihood.unconstrain()
        m.likelihood.variance.set_prior(GPy.priors.Exponential(1))
        m.likelihood.variance.constrain_positive()
        
        
        if self.hmc is None:
            self.hmc = GPy.inference.mcmc.hmc.HMC(m, stepsize=5e-2)
            # Burnin
            self.hmc.sample(num_samples=self.burnin)
        else:
            self.hmc.model = m
        # Start the mcmc chain
        self.mcmc_chain = self.hmc.sample(num_samples=(self.chain_length))
        self.samples = self.mcmc_chain[range(0, self.chain_length, self.chain_length / self.n_hypers)]

        self.models = []
        for sample in self.samples:
            # Instantiate a model for each hyperparam configuration
            kernel = deepcopy(self.kernel)

            for i in range(len(sample) - 1):
                kernel.param_array[i] = sample[i]
            model = GPyModel(kernel, noise_variance=sample[-1])
            model.train(self.X, self.Y, optimize=False)
            self.models.append(model)

    def predict(self, X, full_cov=False):
        if self.models == None:
            logger.error("ERROR: The model needs to be trained first.")
            return None

        mean = np.zeros([self.n_hypers, X.shape[0]])
        var = np.zeros([self.n_hypers, X.shape[0]])

        for i, model in enumerate(self.models):
            m, v = model.predict(X, full_cov)
            mean[i, :] = m
            var[i, :] = v

        return mean.mean(axis=0), var.mean(axis=0)

    def predictive_gradients(self, X):
        if self.models == None:
            logger.error("ERROR: The model needs to be trained first.")
            return None

        mean = np.zeros([self.n_hypers, X.shape[0], X.shape[1], 1])
        var = np.zeros([self.n_hypers, X.shape[0], X.shape[1]])

        for i, model in enumerate(self.models):
            m, v = model.predictive_gradients(X)
            mean[i], = m
            var[i] = v

        return mean.mean(axis=0), var.mean(axis=0)

    def predict_variance(self, X1, X2):
        var = []
        for m in self.models:
            var.append(m.predict_variance(X1, X2))

        return np.array(var)
