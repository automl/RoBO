
import logging
import george
import emcee
import numpy as np

from copy import deepcopy

from robo.models.base_model import BaseModel
from robo.models.gaussian_process import GaussianProcess
from robo.util import normalization

logger = logging.getLogger(__name__)


class GaussianProcessMCMC(BaseModel):

    def __init__(self, kernel, prior=None, n_hypers=20, chain_length=2000,
                 burnin_steps=2000, basis_func=None, dim=None,
                 normalize_output=False, normalize_input=True):
        """
        GaussianProcess model based on the george GP library that uses MCMC
        sampling to marginalise over the hyperparmeters. If you use this class
        make sure that you also use the IntegratedAcqusition function to
        integrate over the GP's hyperparameter as proposed by Snoek et al.

        Parameters
        ----------
        kernel : george kernel object
            Specifies the kernel that is used for all Gaussian Process
        prior : prior object
            Defines a prior for the hyperparameters of the GP. Make sure that
            it implements the Prior interface. During MCMC sampling the
            lnlikelihood is multiplied with the prior.
        n_hypers : int
            The number of hyperparameter samples. This also determines the
            number of walker for MCMC sampling as each walker will
            return one hyperparameter sample.
        chain_length : int
            The length of the MCMC chain. We start n_hypers walker for
            chain_length steps and we use the last sample
            in the chain as a hyperparameter sample.
        burnin_steps : int
            The number of burnin steps before the actual MCMC sampling starts.
        """

        self.kernel = kernel
        if prior is None:
            self.prior = lambda x: 0
        else:
            self.prior = prior
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        self.basis_func = basis_func
        self.dim = dim
        self.models = []
        self.normalize_output = normalize_output
        self.normalize_input = normalize_input
        self.X = None
        self.y = None
        self.is_trained = False

    @BaseModel._check_shapes_train
    def train(self, X, y, do_optimize=True, **kwargs):
        """
        Performs MCMC sampling to sample hyperparameter configurations from the
        likelihood and trains for each sample a GP on X and y

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true we perform MCMC sampling otherwise we just use the
            hyperparameter specified in the kernel.
        """

        if self.normalize_input:
            # Normalize input to be in [0, 1]
            self.X, self.lower, self.upper = normalization.zero_one_normalization(X)
        else:
            self.X = X

        if self.normalize_output:
            # Normalize output to have zero mean and unit standard deviation
            self.y, self.y_mean, self.y_std = normalization.zero_mean_unit_var_normalization(y)
        else:
            self.y = y

        # Use the mean of the data as mean for the GP
        mean = np.mean(self.y, axis=0)
        self.gp = george.GP(self.kernel, mean=mean)

        if do_optimize:
            # We have one walker for each hyperparameter configuration
            sampler = emcee.EnsembleSampler(self.n_hypers,
                                                 len(self.kernel.pars) + 1,
                                                 self.loglikelihood)

            # Do a burn-in in the first iteration
            if not self.burned:
                # Initialize the walkers by sampling from the prior
                self.p0 = self.prior.sample_from_prior(self.n_hypers)
                # Run MCMC sampling
                self.p0, _, _ = sampler.run_mcmc(self.p0,
                                                 self.burnin_steps)

                self.burned = True

            # Start sampling
            pos, _, _ = sampler.run_mcmc(self.p0,
                                         self.chain_length)

            # Save the current position, it will be the start point in
            # the next iteration
            self.p0 = pos

            # Take the last samples from each walker
            self.hypers = sampler.chain[:, -1]

            self.models = []
        else:
            self.hypers = [self.gp.kernel[:]]

        for sample in self.hypers:

            # Instantiate a GP for each hyperparameter configuration
            kernel = deepcopy(self.kernel)
            kernel.pars = np.exp(sample[:-1])
            noise = np.exp(sample[-1])
            model = GaussianProcess(kernel,
                                    basis_func=self.basis_func,
                                    dim=self.dim,
                                    normalize_output=False,
                                    normalize_input=False,
                                    noise=noise)
            model.train(self.X, self.y, do_optimize=False)
            self.models.append(model)

        self.is_trained = True

    def loglikelihood(self, theta):
        """
        Return the loglikelihood (+ the prior) for a hyperparameter
        configuration theta.

        Parameters
        ----------
        theta : np.ndarray(H)
            Hyperparameter vector. Note that all hyperparameter are
            on a log scale.

        Returns
        ----------
        float
            lnlikelihood + prior
        """

        # Bound the hyperparameter space to keep things sane. Note all
        # hyperparameters live on a log scale
        if np.any((-20 > theta) + (theta > 20)):
            return -np.inf
            
        # The last entry is always the noise
        sigma_2 = np.exp(theta[-1])
        # Update the kernel and compute the lnlikelihood.
        self.gp.kernel.pars = np.exp(theta[:-1])
        
        try:
            self.gp.compute(self.X, yerr=np.sqrt(sigma_2))
        except:
            return -np.inf

        return self.prior.lnprob(theta) + self.gp.lnlikelihood(self.y, quiet=True)

    @BaseModel._check_shapes_predict
    def predict(self, X_test, **kwargs):
        r"""
        Returns the predictive mean and variance of the objective function
        at X average over all hyperparameter samples.
        The mean is computed by:
        :math \mu(x) = \frac{1}{M}\sum_{i=1}^{M}\mu_m(x)
        And the variance by:
        :math \sigma^2(x) = (\frac{1}{M}\sum_{i=1}^{M}(\sigma^2_m(x) + \mu_m(x)^2) - \mu^2

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance

        """
        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        if self.normalize_input:
            X_test_norm, _, _ = normalization.zero_one_normalization(X_test, self.lower, self.upper)
        else:
            X_test_norm = X_test

        mu = np.zeros([len(self.models), X_test_norm.shape[0]])
        var = np.zeros([len(self.models), X_test_norm.shape[0]])
        for i, model in enumerate(self.models):
            mu[i], var[i] = model.predict(X_test_norm)

        # See the Algorithm Runtime Prediction paper by Hutter et al.
        # for the derivation of the total variance
        m = mu.mean(axis=0)
        v = np.mean(mu ** 2 + var) - m ** 2

        if self.normalize_output:
            m = normalization.zero_mean_unit_var_unnormalization(m, self.y_mean, self.y_std)

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        return m, v

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        Returns
        ----------
        incumbent: ndarray (D,)
            current incumbent
        incumbent_value: ndarray (N,)
            the observed value of the incumbent
        """
        inc, inc_value = super(GaussianProcessMCMC, self).get_incumbent()
        if self.normalize_input:
            inc = normalization.zero_one_unnormalization(inc, self.lower, self.upper)

        if self.normalize_output:
            inc_value = normalization.zero_mean_unit_var_unnormalization(inc_value, self.y_mean, self.y_std)

        return inc, inc_value
