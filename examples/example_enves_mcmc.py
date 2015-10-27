'''
Created on Jun 26, 2015

@author: Aaron Klein
'''

import setup_logger

import numpy as np
import george


from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.models.gaussian_process import GaussianProcess
from robo.acquisition.env_entropy_search import EnvEntropySearch
from robo.acquisition.integrated_acquisition import IntegratedAcquisition
from robo.maximizers.direct import Direct
from robo.recommendation.incumbent import compute_incumbent
from robo.recommendation.optimize_posterior import env_optimize_posterior_mean_and_std
from robo.task.env_branin import EnvBranin
from robo.solver.environment_search import EnvironmentSearch


def lnprior(x):
    if np.any((-10 > x) + (x > 10)):
        return -np.inf 
    return 0

branin = EnvBranin()

noise = 1.0
cov_amp = 2
exp_kernel = george.kernels.ExpSquaredKernel([1.0, 1.0, 1.0], ndim=3)
noise_kernel = george.kernels.WhiteKernel(noise, ndim=3)
kernel = cov_amp * (exp_kernel + noise_kernel)

model = GaussianProcessMCMC(kernel, lnprior=lnprior, chain_length=100, burnin_steps=200, n_hypers=20)


cost_noise = 1.0
cost_cov_amp = 2
cost_exp_kernel = george.kernels.ExpSquaredKernel([1.0, 1.0, 1.0], ndim=3)
cost_noise_kernel = george.kernels.WhiteKernel(cost_noise, ndim=3)
cost_kernel = cost_cov_amp * (cost_exp_kernel + cost_noise_kernel)

cost_model = GaussianProcessMCMC(cost_kernel, lnprior=lnprior)

es = EnvEntropySearch(model, cost_model, branin.X_lower, branin.X_upper, env_optimize_posterior_mean_and_std, branin.is_env, 50)
acquisition_func = IntegratedAcquisition(model, es, cost_model)
        

maximizer = Direct(acquisition_func, branin.X_lower, branin.X_upper)
bo = EnvironmentSearch(acquisition_func=acquisition_func,
                  model=model,
                  cost_model=cost_model,
                  maximize_func=maximizer,
                  task=branin,
                  synthetic_func=True)
bo.run(10)
