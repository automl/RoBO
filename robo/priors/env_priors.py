
import numpy as np
from robo.priors.base_prior import BasePrior, TophatPrior
from robo.priors.base_prior import LognormalPrior, HorseshoePrior, NormalPrior

from scipy.stats import norm


class EnvPrior(BasePrior):

    def __init__(self, n_dims, n_ls, n_lr, rng=None):

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        # The number of hyperparameters
        self.n_dims = n_dims

        # The number of lengthscales
        self.n_ls = n_ls

        # The number of params of the bayes linear reg kernel
        self.n_lr = n_lr
        self.bayes_lin_prior = NormalPrior(sigma=1, mean=0, rng=self.rng)

        # Prior for the Matern52 lengthscales
        self.tophat = TophatPrior(-10, 2, rng=self.rng)

        # Prior for the covariance amplitude
        self.ln_prior = LognormalPrior(mean=-2, sigma=1.0, rng=self.rng)

        # Prior for the noise
        self.horseshoe = HorseshoePrior(scale=0.001, rng=self.rng)

    def lnprob(self, theta):

        lp = 0

        # Covariance amplitude
        lp += self.ln_prior.lnprob(theta[0])

        # Lengthscales
        lp += self.tophat.lnprob(theta[1:self.n_ls + 1])

        # Prior for the Bayesian regression kernel
        pos = (self.n_ls + 1)
        end = (self.n_ls + self.n_lr + 1)

        for t in theta[pos:end]:
            lp += self.bayes_lin_prior.lnprob(t)

        # Noise
        lp += self.horseshoe.lnprob(theta[-1])

        return lp

    def sample_from_prior(self, n_samples):
        p0 = np.zeros([n_samples, self.n_dims])

        # Covariance amplitude
        p0[:, 0] = self.ln_prior.sample_from_prior(n_samples)[:, 0]

        # Lengthscales
        ls_sample = np.array([self.tophat.sample_from_prior(n_samples)[:, 0]
                              for _ in range(0, self.n_ls)]).T
        p0[:, 1:(self.n_ls + 1)] = ls_sample

        # Bayesian linear regression
        pos = (self.n_ls + 1)
        end = (self.n_ls + self.n_lr + 1)

        samples = np.array([self.bayes_lin_prior.sample_from_prior(n_samples)[:, 0]
                                   for _ in range(0, self.n_lr)]).T

        p0[:, pos:end] = samples
        # Noise
        p0[:, -1] = self.horseshoe.sample_from_prior(n_samples)[:, 0]

        return p0


class EnvNoisePrior(BasePrior):

    def __init__(self, n_dims, n_ls, n_lr):

        # The number of hyperparameters
        self.n_dims = n_dims

        # The number of lengthscales
        self.n_ls = n_ls

        # The number of params of the bayes linear reg kernel
        self.n_lr = n_lr

        # Prior for the Matern52 lengthscales
        self.tophat = TophatPrior(-2, 2)

        # Prior for the covariance amplitude
        self.ln_prior = LognormalPrior(mean=0.0, sigma=1.0)

        # Prior for the noise
        self.horseshoe = HorseshoePrior(scale=0.1)

    def lnprob(self, theta):

        lp = 0

        # Covariance amplitude
        lp += self.ln_prior.lnprob(theta[0])

        # Lengthscales
        lp += self.tophat.lnprob(theta[1:self.n_ls + 1])

        # Prior for the Bayesian regression kernel
        pos = (self.n_ls + 1)
        end = (self.n_ls + self.n_lr + 1)
        lp += -np.sum((theta[pos:end]) ** 2 / 10.)

        # alpha
        lp += norm.pdf(theta[end], loc=-7, scale=1)
        # beta
        lp += norm.pdf(theta[end + 1], loc=0.5, scale=1)

        # Noise
        lp += self.horseshoe.lnprob(theta[-1])

        return lp

    def sample_from_prior(self, n_samples):
        p0 = np.zeros([n_samples, self.n_dims])

        # Covariance amplitude
        p0[:, 0] = self.ln_prior.sample_from_prior(n_samples)[:, 0]

        # Lengthscales
        ls_sample = np.array([self.tophat.sample_from_prior(n_samples)[:, 0]
                              for _ in range(0, self.n_ls)]).T
        p0[:, 1:(self.n_ls + 1)] = ls_sample

        # Bayesian linear regression
        pos = (self.n_ls + 1)
        end = (self.n_ls + self.n_lr + 1)
        p0[:, pos:end] = np.array([5 * np.random.randn(n_samples) - 5
                                   for _ in range(0, self.n_lr)]).T

        # Env Noise
        p0[:, end] = np.random.randn(n_samples) - 7
        p0[:, end + 1] = np.random.randn(n_samples) + 0.5

        # Noise
        p0[:, -1] = self.horseshoe.sample_from_prior(n_samples)[:, 0]

        return p0


class MTBOPrior(BasePrior):

    def __init__(self, n_dims, n_ls, n_kt, rng=None):

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        # The number of hyperparameters
        self.n_dims = n_dims

        # The number of lengthscales
        self.n_ls = n_ls

        # The number of entries of
        # kernel matrix for the tasks
        self.n_kt = n_kt

        # Prior for the Matern52 lengthscales
        self.tophat = TophatPrior(-10, 2, rng=self.rng)

        # Prior for the covariance amplitude
        self.ln_prior = LognormalPrior(mean=0.0, sigma=1.0, rng=self.rng)

        # Prior for the noise
        self.horseshoe = HorseshoePrior(scale=0.1, rng=self.rng)

        self.tophat_task = TophatPrior(-1, 0, rng=self.rng)

    def lnprob(self, theta):

        lp = 0

        # Covariance amplitude
        lp += self.ln_prior.lnprob(theta[0])

        # Lengthscales
        lp += self.tophat.lnprob(theta[1:self.n_ls + 1])

        # Prior for the task kernel
        task_chol = theta[self.n_ls + 1:self.n_ls + 1 + self.n_kt]
        # lp -= np.sum(0.5 * (task_chol / 1) ** 2)
        lp += self.tophat_task.lnprob(task_chol)

        # Noise
        lp += self.horseshoe.lnprob(theta[-1])

        return lp

    def sample_from_prior(self, n_samples):
        p0 = np.zeros([n_samples, self.n_dims])

        # Covariance amplitude
        p0[:, 0] = self.ln_prior.sample_from_prior(n_samples)[:, 0]

        # Lengthscales
        ls_sample = np.array([self.tophat.sample_from_prior(n_samples)[:, 0]
                              for _ in range(0, self.n_ls)]).T
        p0[:, 1:(self.n_ls + 1)] = ls_sample

        # Task Kernel
        pos = (self.n_ls + 1)
        end = (self.n_ls + self.n_kt + 1)
        # p0[:, pos:end] = self.rng.randn(n_samples, end - pos)
        p0[:, pos:end] = np.array([self.tophat_task.sample_from_prior(n_samples) for i in range(0, end - pos)]).T

        # Noise
        p0[:, -1] = self.horseshoe.sample_from_prior(n_samples)[:, 0]

        return p0
