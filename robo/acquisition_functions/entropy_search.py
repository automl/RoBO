import numpy as np
import emcee
import logging
import time
from scipy.stats import norm

from robo.acquisition_functions.entropy import Entropy
from robo.recommendation.optimize_posterior import optimize_posterior_mean_and_std

sq2 = np.sqrt(2)
l2p = np.log(2) + np.log(np.pi)
eps = np.finfo(np.float32).eps

logger = logging.getLogger(__name__)


class EntropySearch(Entropy):
    def __init__(self, model, X_lower, X_upper,
                 compute_inc=optimize_posterior_mean_and_std,
                 Nb=50, Nf=1000,
                 sampling_acquisition=None,
                 sampling_acquisition_kw={"par": 0.0},
                 Np=300, **kwargs):

        super(EntropySearch, self).__init__(model, X_lower, X_upper, Nb,
                                        compute_inc, sampling_acquisition,
                                        sampling_acquisition_kw, Np, **kwargs)
        self.Nf = Nf
        self.Np = Np

    def compute(self, X, *args):
        return self.dh_fun(X)

    def update(self, model):
        self.model = model

        self.sn2 = self.model.get_noise()

        # Sample representer points
        self.sampling_acquisition.update(model)
        self.update_representer_points()

        # Omega values which are needed for the innovations
        # by sampling from a uniform grid
        self.W = norm.ppf(np.linspace(1. / (self.Np + 1),
                                    1 - 1. / (self.Np + 1),
                                    self.Np))[np.newaxis, :]

        # Compute current posterior belief at the representer points
        self.Mb, self.Vb = self.model.predict(self.zb, full_cov=True)

        # Random samples that are used for drawing functions from the GP
        self.F = np.random.multivariate_normal(mean=np.zeros(self.Nb),
                                               cov=np.eye(self.Nb),
                                               size=self.Nf)

        # Compute the current pmin at the representer points
        self.pmin = self.compute_pmin(self.Mb[:, np.newaxis], self.Vb)
        self.logP = np.log(self.pmin)

        self.update_best_guesses()

        self.plotting = True

    def compute_pmin(self, m, V):
        """
        Computes the distribution over the global minimum based on
        functions drawn from a posterior distribution

        Parameters
        ----------

        Returns
        -------
        pmin: np.array(Nb)
            Probability that the corresponding representer point is
            the minimum

        """
        noise = 0
        while(True):
            try:
                cV = np.linalg.cholesky(V + noise * np.eye(V.shape[0]))
                break
            except np.linalg.LinAlgError:
                embed()
                if noise == 0:
                    noise = 1e-10
                if noise == 10:
                    raise np.linalg.LinAlgError('Cholesky '
                        'decomposition failed.')
                else:
                    noise *= 10
                logger.error("Cholesky decomposition failed."
                              "Add %f noise on the diagonal." % noise)

        # Draw new function samples from the innovated GP
        # on the representer points
        funcs = np.dot(cV, self.F.T)
        funcs = funcs[:, :, None]
        m = m[:, None, :]
        funcs = m + funcs

        funcs = funcs.reshape(funcs.shape[0], funcs.shape[1] * funcs.shape[2])

        # Determine the minima for each function sample
        mins = np.argmin(funcs, axis=0)
        c = np.bincount(mins)

        # Count how often each representer point was the minimum
        min_count = np.zeros((self.Nb,))
        min_count[:len(c)] += c
        pmin = (min_count / funcs.shape[1])[:, None]
        pmin[np.where(pmin < 1e-70)] = 1e-70

        return pmin

    def innovations(self, x, rep):
        # Get the variance at x with noise
        _, v = self.model.predict(x)

        # Get the variance at x without noise
        v_ = v - self.sn2

        # Compute the variance between the test point x and the representers
        sigma_x_rep = self.model.predict_variance(rep, x)

        norm_cov = np.dot(sigma_x_rep, np.linalg.inv(v_))

        # Compute the stochastic innovation for the mean
        try:
            dm_rep = np.dot(norm_cov,
                    np.linalg.cholesky(np.array([v])))[:, :, 0]
        except:
            embed()
        dm_rep = dm_rep.dot(self.W)

        # Compute the deterministic innovation for the variance
        dv_rep = -norm_cov.dot(sigma_x_rep.T)

        return dm_rep, dv_rep

    def change_pmin_by_innovation(self, x):

        # Compute the change of our posterior at the representer points for
        # different halluzinate function values of x
        dmdb, dvdb = self.innovations(x, self.zb)

        # Update mean and variance of the posterior (at the representer points)
        # by the innovations
        Mb_new = self.Mb[:, None] + dmdb
        Vb_new = self.Vb + dvdb

        # Return the fantasized pmin
        return self.compute_pmin(Mb_new, Vb_new)

    def dh_fun(self, x):
        if x.shape[0] > 1:
            raise ValueError("EntropyMC is only for single test points")
        # Compute the fantasized pmin if we would evaluate at x
        new_pmin = self.change_pmin_by_innovation(x)

        # Calculate the Kullback-Leibler divergence between the old and the
        # fantasised pmin
        H_old = np.sum(np.multiply(self.pmin, (self.logP + self.lmb)))
        H_new = np.sum(np.multiply(new_pmin, (np.log(new_pmin) + self.lmb)))

        # Return the expected information gain
        return np.array([[-H_new + H_old]])
