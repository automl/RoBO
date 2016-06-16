import george

from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.acquisition.ei import EI
from robo.maximizers.direct import Direct
from robo.task.synthetic_functions.branin import Branin
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.priors.default_priors import DefaultPrior
from robo.acquisition.integrated_acquisition import IntegratedAcquisition


task = Branin()

noise = 1.0
cov_amp = 2
exp_kernel = george.kernels.Matern52Kernel([1.0, 1.0], ndim=2)
kernel = cov_amp * exp_kernel

prior = DefaultPrior(len(kernel) + 1)
model = GaussianProcessMCMC(kernel, prior=prior,
                            chain_length=100, burnin_steps=200, n_hypers=20)

ei = EI(model, task.X_lower, task.X_upper)
acquisition_func = IntegratedAcquisition(model, ei, task.X_lower, task.X_upper)

maximizer = Direct(acquisition_func, task.X_lower, task.X_upper)

bo = BayesianOptimization(acquisition_func=acquisition_func,
                          model=model,
                          maximize_func=maximizer,
                          task=task)

print bo.run(10)
