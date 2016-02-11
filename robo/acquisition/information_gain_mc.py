import numpy as np
import emcee
import logging
from scipy.stats import norm

from robo.acquisition.log_ei import LogEI
from robo.acquisition.base import AcquisitionFunction
from robo.initial_design.init_random_uniform import init_random_uniform

logger = logging.getLogger(__name__)


class InformationGainMC(AcquisitionFunction):
    def __init__(self, model, X_lower, X_upper,
                 Nb=50, Nf=500,
                 sampling_acquisition=None,
                 sampling_acquisition_kw={"par": 0.0},
                 Np=50, **kwargs):

        """
        The InformationGainMC computes the asymptotically exact, sampling
        based variant of the entropy search acquisition function [1] by
        approximating the distribution over the minimum with MC sampling.

        [1] Hennig and C. J. Schuler
            Entropy search for information-efficient global optimization
            Journal of Machine Learning Research, 13, 2012


        Parameters
        ----------
        model: Model object
            A model should have following methods:
            - predict(X)
            - predict_variance(X1, X2)
        X_lower: np.ndarray (D)
            Lower bounds of the input space
        X_upper: np.ndarray (D)
            Upper bounds of the input space
        Nb: int
            Number of representer points.
        Np: int
            Number of prediction points at X to calculate stochastic changes
            of the mean for the representer points
        Nf: int
            Number of functions that are sampled to approximate pmin
        sampling_acquisition: AcquisitionFunction
            A function to be used in calculating the density that
            representer points are to be sampled from. It uses
        sampling_acquisition_kw: dict
            Additional keyword parameters to be passed to sampling_acquisition

        """

        self.Nb = Nb
        super(InformationGainMC, self).__init__(model, X_lower, X_upper)
        self.D = self.X_lower.shape[0]
        self.sn2 = None
        if sampling_acquisition is None:
            sampling_acquisition = LogEI
        self.sampling_acquisition = sampling_acquisition(
            model, self.X_lower, self.X_upper, **sampling_acquisition_kw)
        self.Nf = Nf
        self.Np = Np

    def compute(self, X, derivative=False, *args):

        if derivative:
            raise NotImplementedError
        # Compute the fantasized pmin if we would evaluate at x
        new_pmin = self.change_pmin_by_innovation(X)

        # Calculate the Kullback-Leibler divergence between the old and the
        # fantasised pmin
        H = -np.sum(np.multiply(np.exp(self.logP), (self.logP + self.lmb)))
        dHp = - np.sum(np.multiply(new_pmin,
                            np.add(np.log(new_pmin), self.lmb)), axis=0) - H
        # We maximize
        return -np.array([dHp])

    def sample_representer_points(self):
        self.sampling_acquisition.update(self.model)

        start_points = init_random_uniform(self.X_lower,
                                       self.X_upper,
                                       self.Nb)

        sampler = emcee.EnsembleSampler(self.Nb,
                                        self.D,
                                        self.sampling_acquisition_wrapper)

        # zb are the representer points and lmb are their log EI values
        self.zb, self.lmb, _ = sampler.run_mcmc(start_points, 200)

        if len(self.zb.shape) == 1:
            self.zb = self.zb[:, None]
        if len(self.lmb.shape) == 1:
            self.lmb = self.lmb[:, None]

    def sampling_acquisition_wrapper(self, x):
        if np.any(x < self.X_lower) or np.any(x > self.X_upper):
            return -np.inf
        return self.sampling_acquisition(np.array([x]))[0]

    def update(self, model):
        self.model = model

        self.sn2 = self.model.get_noise()

        # Sample representer points
        self.sampling_acquisition.update(model)
        self.sample_representer_points()

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
        self.pmin = self.compute_pmin(self.Mb, self.Vb)
        self.logP = np.log(self.pmin)

    def compute_pmin(self, m, V):
        """
        Computes the distribution over the global minimum based on
        functions drawn from a posterior distribution

        Parameters
        ----------
        m: np.array(Nb, Np)
            Mean of the representer point
        V: np.array(Nb, Nb)
            Variance of the representer point
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

                if noise == 0:
                    noise = 1e-10
                if noise == 10000:
                    raise np.linalg.LinAlgError('Cholesky '
                        'decomposition failed.')
                else:
                    noise *= 10

        if noise > 0:
            logger.error("Add %f noise on the diagonal." % noise)
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
        pmin = (min_count / funcs.shape[1])
        pmin[np.where(pmin < 1e-70)] = 1e-70

        return pmin

    def innovations(self, x, rep):
        # Get the variance at x with noise
        _, v = self.model.predict(x)

        # Get the variance at x without noise
        v_ = v - self.sn2

        # Compute the variance between the test point x and
        # the representer points
        sigma_x_rep = self.model.predict_variance(rep, x)

        norm_cov = np.dot(sigma_x_rep, np.linalg.inv(v_))
        # Compute the stochastic innovation for the mean
        dm_rep = np.dot(norm_cov,
                    np.linalg.cholesky(v + 1e-10))
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
        Mb_new = self.Mb + dmdb
        Vb_new = self.Vb + dvdb

        # Return the fantasized pmin
        return self.compute_pmin(Mb_new, Vb_new)
