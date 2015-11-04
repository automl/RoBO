'''
Created on Oct 27, 2015

@author: Aaron Klein
'''
import george
import cma
import numpy as np

from robo.task.branin import Branin
from robo.priors.base_prior import BasePrior
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.priors import default_priors
from robo.acquisition.ei import EI
from robo.maximizers.direct import Direct
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.recommendation.incumbent import compute_incumbent
from robo.task.noise_task import NoiseTask


class MyPrior(BasePrior):

    def __init__(self, n_dims):
        super(MyPrior, self).__init__()
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


def global_optimize_posterior(model, X_lower, X_upper, startpoint):
    def f(x):
        mu, var = model.predict(x[np.newaxis, :])
        return (mu + np.sqrt(var))[0, 0]
    # Use CMAES to optimize the posterior mean + std
    res = cma.fmin(f, startpoint, 0.6, options={"bounds": [X_lower, X_upper]})
    return res[0], np.array([res[1]])


burnin = 100
chain_length = 200
n_hypers = 20

task = Branin()

cov_amp = 1.0
config_kernel = george.kernels.Matern52Kernel(np.ones([task.n_dims]) * 0.5,
                                               ndim=task.n_dims)

noise_kernel = george.kernels.WhiteKernel(0.01, ndim=task.n_dims)
kernel = cov_amp * (config_kernel + noise_kernel)

prior = MyPrior(len(kernel))

model = GaussianProcessMCMC(kernel, prior=prior, burnin=burnin,
                            chain_length=chain_length, n_hypers=n_hypers)

acquisition_func = EI(model, X_upper=task.X_upper, X_lower=task.X_lower,
                      compute_incumbent=compute_incumbent, par=0.1)

maximizer = Direct(acquisition_func, task.X_lower, task.X_upper)

bo = BayesianOptimization(acquisition_func=acquisition_func,
                          model=model,
                          maximize_func=maximizer,
                          task=task,
                          recommendation_strategy=global_optimize_posterior)

bo.run(20)


