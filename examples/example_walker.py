import george
import numpy as np

from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.acquisition.ei import EI
from robo.maximizers.direct import Direct
from robo.task.controlling_tasks.walker import Walker
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.priors.default_priors import DefaultPrior
from robo.acquisition.integrated_acquisition import IntegratedAcquisition



task = Walker()
test = '/test'

kernel = george.kernels.Matern52Kernel(np.ones([task.n_dims]),ndim=task.n_dims)
prior = DefaultPrior(len(kernel))
model = GaussianProcessMCMC(kernel, prior=prior,
                            chain_length=100, burnin_steps=200, n_hypers=8)

ei = EI(model, task.X_lower, task.X_upper)
acquisition_func = IntegratedAcquisition(model, ei, task.X_lower, task.X_upper)

maximizer = Direct(acquisition_func, task.X_lower, task.X_upper)

bo = BayesianOptimization(acquisition_func=acquisition_func,
                          model=model,
                          maximize_func=maximizer,
                          task=task,
                          save_dir = test)

print bo.run(2)