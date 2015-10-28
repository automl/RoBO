'''
Created on Jun 26, 2015

@author: Aaron Klein
'''

import setup_logger

import numpy as np
import george


from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.models.gaussian_process import GaussianProcess
from robo.acquisition.environment_entropy import EnvironmentEntropy
from robo.acquisition.integrated_acquisition import IntegratedAcquisition
from robo.maximizers.direct import Direct
from robo.recommendation.optimize_posterior import env_optimize_posterior_mean_and_std
from robo.task.branin import Branin
from robo.task.environmental_synthetic_function import EnvironmentalSyntheticFunction
from robo.solver.environment_search import EnvironmentSearch
from robo.priors import default_priors
from robo.priors.base_prior import BasePrior


class Prior(BasePrior):

    def __init__(self, n_dims):

        # The number of hyperparameters
        self.n_dims = n_dims

        # Prior for the Matern52 lengthscales
        self.tophat = default_priors.TophatPrior(-2, 2)

        # Prior for the covariance amplitude
        self.ln_prior = default_priors.LognormalPrior(mean=0.0, sigma=1.0)

        # Prior for the noise
        self.horseshoe = default_priors.HorseshoePrior(scale=0.1)

    def lnprob(self, theta):
        lp = 0
        # Covariance amplitude
        lp += self.ln_prior.lnprob(theta[0])
        # Lengthscales
        lp += self.tophat.lnprob(theta[1:-1])
        # Noise
        lp += self.horseshoe.lnprob(theta[-1])

        return lp

    def sample_from_prior(self, n_samples):
        p0 = np.zeros([n_samples, self.n_dims])
        # Covariance amplitude
        p0[:, 0] = self.ln_prior.sample_from_prior(n_samples)
        # Lengthscales
        ls_sample = np.array([self.tophat.sample_from_prior(n_samples)
                              for _ in range(1, (self.n_dims - 1))]).T
        p0[:, 1:(self.n_dims - 1)] = ls_sample
        # Noise
        p0[:, -1] = self.horseshoe.sample_from_prior(n_samples)

        return p0

# Warp the original Branin function to an the system size
# as additional input and an exponential cost function
branin = Branin()
task = EnvironmentalSyntheticFunction(branin)

# Define the kernel + prior for modeling the objective function
noise = 1.0
cov_amp = 2
exp_kernel = george.kernels.ExpSquaredKernel([1.0, 1.0, 1.0], ndim=3)
noise_kernel = george.kernels.WhiteKernel(noise, ndim=3)
kernel = cov_amp * (exp_kernel + noise_kernel)

prior = Prior(len(kernel))

model = GaussianProcessMCMC(kernel, prior=prior, chain_length=100,
                            burnin_steps=200, n_hypers=20)

# Define the kernel + prior for modeling the cost function
cost_noise = 1.0
cost_cov_amp = 2
cost_exp_kernel = george.kernels.ExpSquaredKernel([1.0, 1.0, 1.0], ndim=3)
cost_noise_kernel = george.kernels.WhiteKernel(cost_noise, ndim=3)
cost_kernel = cost_cov_amp * (cost_exp_kernel + cost_noise_kernel)

cost_prior = Prior(len(cost_kernel))

cost_model = GaussianProcessMCMC(cost_kernel, prior=cost_prior)

# Inititalize the BO ingredients
es = EnvironmentEntropy(model, cost_model, task.X_lower, task.X_upper,
                        env_optimize_posterior_mean_and_std, task.is_env, 50)
acquisition_func = IntegratedAcquisition(model, es, cost_model)
maximizer = Direct(acquisition_func, task.X_lower, task.X_upper)

bo = EnvironmentSearch(acquisition_func=acquisition_func,
                  model=model,
                  cost_model=cost_model,
                  maximize_func=maximizer,
                  task=task,
                  n_init_points=2)
bo.run(20)
