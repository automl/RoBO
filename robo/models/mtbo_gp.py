import george
import emcee
import numpy as np

from copy import deepcopy

from robo.util import normalization
from robo.models.gaussian_process import GaussianProcess
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC


def normalize(X, lower, upper):
    X_norm, _, _ = normalization.zero_one_normalization(X[:, :-1], lower, upper)
    X_norm = np.concatenate((X_norm,  np.rint(X[:, None, -1])), axis=1)
    return X_norm


class MTBOGPMCMC(GaussianProcessMCMC):
    def __init__(self, kernel,
                 prior=None, n_hypers=20,
                 chain_length=2000, burnin_steps=2000,
                 normalize_output=False,
                 rng=None,
                 lower=None,
                 upper=None,
                 noise=-8):

        self.hypers = None
        super(MTBOGPMCMC, self).__init__(kernel, prior,
                                            n_hypers, chain_length,
                                            burnin_steps,
                                            normalize_output=normalize_output,
                                            normalize_input=False,
                                            rng=rng, lower=lower,
                                            upper=upper, noise=noise)

    def train(self, X, y, do_optimize=True, **kwargs):
        self.X = normalize(X, self.lower, self.upper)

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
                if self.prior is None:
                    self.p0 = np.random.rand(self.n_hypers, len(self.kernel.pars) + 1)
                else:
                    self.p0 = self.prior.sample_from_prior(self.n_hypers)
                # Run MCMC sampling
                self.p0, _, _ = sampler.run_mcmc(self.p0,
                                                 self.burnin_steps,
                                                 rstate0=self.rng)

                self.burned = True

            # Start sampling
            pos, _, _ = sampler.run_mcmc(self.p0,
                                         self.chain_length,
                                         rstate0=self.rng)

            # Save the current position, it will be the start point in
            # the next iteration
            self.p0 = pos

            # Take the last samples from each walker
            self.hypers = sampler.chain[:, -1]

        else:
            if self.hypers is None:
                self.hypers = self.gp.kernel[:].tolist()
                self.hypers.append(self.noise)
                self.hypers = [self.hypers]

        self.models = []
        for sample in self.hypers:

            # Instantiate a GP for each hyperparameter configuration
            kernel = deepcopy(self.kernel)
            #kernel.pars = np.exp(sample[:-1])
            kernel.vector = sample[:-1]
            noise = np.exp(sample[-1])
            model = MTBOGP(kernel,
                              normalize_output=self.normalize_output,
                              noise=noise,
                              lower=self.lower,
                              upper=self.upper,
                              rng=self.rng)
            model.train(X, y, do_optimize=False)
            self.models.append(model)

        self.is_trained = True


class MTBOGP(GaussianProcess):
    def __init__(self, kernel, prior=None,
                 noise=1e-3, use_gradients=False,
                 normalize_output=False,
                 lower=None, upper=None, rng=None):
        super(MTBOGP, self).__init__(kernel=kernel,
                                     prior=prior,
                                     noise=noise,
                                     use_gradients=use_gradients,
                                     normalize_output=normalize_output,
                                     normalize_input=False,
                                     lower=lower,
                                     upper=upper,
                                     rng=rng)

    def train(self, X, y, do_optimize=True):
        self.original_X = X
        X_norm = normalize(X, self.lower, self.upper)
        return super(MTBOGP, self).train(X_norm, y, do_optimize)

    def predict(self, X_test, full_cov=False, **kwargs):
        X_norm = normalize(X_test, self.lower, self.upper)
        return super(MTBOGP, self).predict(X_norm, full_cov)

    def sample_functions(self, X_test, n_funcs=1):
        X_norm = normalize(X_test, self.lower, self.upper)
        return super(MTBOGP, self).sample_functions(X_norm, n_funcs)

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

        projection = np.ones([self.original_X.shape[0], 1]) * 1

        X_projected = np.concatenate((self.original_X[:, :-1], projection), axis=1)
        X_norm = normalize(X_projected, self.lower, self.upper)

        m, _ = self.predict(X_norm)

        best = np.argmin(m)
        incumbent = X_projected[best]
        incumbent_value = m[best]

        return incumbent, incumbent_value
