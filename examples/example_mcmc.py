from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.acquisition.entropy import Entropy
from robo.acquisition.ei import EI
from robo.maximizers.direct import Direct
from robo.recommendation.incumbent import compute_incumbent
from robo.task.branin import Branin
from robo.solver.bayesian_optimization import BayesianOptimization
from george.kernels import ExpSquaredKernel
from robo.acquisition.integrated_acquisition import IntegratedAcquisition
from robo.recommendation.optimize_posterior import optimize_posterior_mean_and_std, env_optimize_posterior_mean_and_std

import logging
logging.basicConfig(level=logging.DEBUG)

task = Branin()

noise = 1.0
cov_amp = 2
exp_kernel = george.kernels.ExpSquaredKernel([1.0, 1.0], ndim=2)
noise_kernel = george.kernels.WhiteKernel(noise, ndim=2)
kernel = cov_amp * (exp_kernel + noise_kernel)

model = GaussianProcessMCMC(kernel, chain_length=100, burnin_steps=20)

#entropy = Entropy(model, task.X_lower, task.X_upper, 50, optimize_posterior_mean_and_std)
#acquisition_func = IntegratedAcquisition(model, entropy)

ei = EI(model, task.X_lower, task.X_upper, compute_incumbent)
acquisition_func = IntegratedAcquisition(model, ei)

maximizer = Direct(acquisition_func, task.X_lower, task.X_upper)

bo = BayesianOptimization(acquisition_func=acquisition_func,
                          model=model,
                          maximize_func=maximizer,
                          task=task,
                          save_dir="/home/kleinaa/temp",
                          num_save=1)

print bo.run(10)