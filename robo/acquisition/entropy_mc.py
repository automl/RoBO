import numpy as np
import emcee
import logging
from scipy.stats import norm
from robo.acquisition.log_ei import LogEI
from robo.acquisition.entropy import Entropy
from robo.recommendation.optimize_posterior import optimize_posterior_mean_and_std

sq2 = np.sqrt(2)
l2p = np.log(2) + np.log(np.pi)
eps = np.finfo(np.float32).eps

logger = logging.getLogger(__name__)

class EntropyMC(Entropy):
    """
    The EntropyMC contains the asymptotically exact, sampling based variant of the entropy search acquisition function.

    :param model: A model should have following methods:

        - predict(X)
        - predict_variance(X1, X2)
    :param X_lower: Lower bounds for the search, its shape should be 1xD (D = dimension of search space)
    :type X_lower: np.ndarray (1,D)
    :param X_upper: Upper bounds for the search, its shape should be 1xD (D = dimension of search space)
    :type X_upper: np.ndarray (1,D)
    :param Nb: Number of representer points.
    :type Nb: int
    :param sampling_acquisition: A function to be used in calculating the density that representer points are to be sampled from. It uses
    :type samping_acquisition: AcquisitionFunction
    :param sampling_acquisition_kw: Additional keyword parameters to be passed to sampling_acquisition
    :type sampling_acquisition_kw: dict
    :param Np: Number of prediction points at X to calculate stochastic changes of the mean for the representer points 
    :type Np: int
    :param Nf: Number of functions to be sampled.
    :type Nf: int
    :param loss_function: The loss function to be used in the calculation of the entropy. If not specified it default is log loss (cf. loss_functions module).

    """
    def __init__(self, model, X_lower, X_upper, compute_inc=optimize_posterior_mean_and_std, Nb=50, Nf=1000, sampling_acquisition=None, sampling_acquisition_kw={"par": 0.0}, Np=300, **kwargs):
        super(EntropyMC, self).__init__(model, X_lower, X_upper, Nb, compute_inc, sampling_acquisition, sampling_acquisition_kw, Np, **kwargs)
        self.Nf = Nf
        self.Np = Np

    def compute(self, X, derivative=False, **kwargs):
        """
        :param X: The point at which the function is to be evaluated. Its shape is (1,D), where n is the dimension of the search space.
        :type X: np.ndarray (1, n)
        :param derivative: Controls whether the derivative is calculated and returned.
        :type derivative: Boolean
        :return: The expected difference of the loss function at X and optionally its derivative.
        :rtype: np.ndarray(1, 1) or (np.ndarray(1, 1), np.ndarray(1, D)).
        :raises BayesianOptimizationError: if X.shape[0] > 1. Only single X can be evaluated.
        """
        if derivative:
            logger.error("EntropyMC does not support derivative calculation until now")
            return
        return self.dh_fun(X)

    def update(self, model):
        super(EntropyMC, self).update(model)
        self.sn2 = self._get_noise()

        self.sampling_acquisition.update(model)
        self.update_representer_points()
        # Omega values which are needed for the innovations
        # Draw W from a normal distribution
        #self.W = np.random.randn(1, self.Np)

        # Estimate W by a uniform grid
        self.W = norm.ppf(np.linspace(1. / (self.Np + 1),
                                    1 - 1. / (self.Np + 1),
                                    self.Np))[np.newaxis, :]

        self.Mb, self.Vb = self.model.predict(self.zb, full_cov=True)
        # Draw random number for the hallucinated values they have to be the same for each innovation
        self.F = np.random.multivariate_normal(mean=np.zeros(self.Nb), cov=np.eye(self.Nb), size=self.Nf)
        if np.any(np.isnan(self.Vb)):
            raise Exception(self.Vb)
        try:
            self.cVb = np.linalg.cholesky(self.Vb)

        except np.linalg.LinAlgError:
            self.cVb = np.linalg.cholesky(self.Vb + 1e-10 * np.eye(self.Vb.shape[0]))

        # Draw function values on the representer points based on the current mean / variance of the GP and the random numbers from above
        self.f = np.add(np.dot(self.cVb, self.F.T).T, self.Mb).T
        # Compute the current pmin
        self.pmin = self.calc_pmin(self.f)
        self.logP = np.log(self.pmin)
        self.update_best_guesses()

    def calc_pmin(self, f):
        logger.debug(f.shape)
        if len(f.shape) == 3:
            f = f.reshape(f.shape[0], f.shape[1] * f.shape[2])
        # Determine the minima for each function sample
        mins = np.argmin(f, axis=0)
        c = np.bincount(mins)
        # Count how often each representer point was the minimum
        min_count = np.zeros((self.Nb,))
        min_count[:len(c)] += c
        pmin = (min_count / f.shape[1])[:, None]
        pmin[np.where(pmin < 1e-70)] = 1e-70
        return pmin

    def change_pmin_by_innovation(self, x):
        Lx, s, v = self._gp_innovation_local(x)
        dMdb = Lx / s * np.sqrt(v)  # Sigma(xstar, zb) * (1 / sigma(x)^2) * Cholesky(Sigma(x,x) + noise)
        dVdb = -Lx.dot(Lx.T)  # Sigma(x, zb) * (1 / sigma(x)^2) * Sigma(zb, x)
        # Add the stochastic factor W to the innovations
        stoch_changes = dMdb.dot(self.W)
        # Update mean and variance of the posterior (at the representer points) by the innovations
        Mb_new = self.Mb[:, None] + stoch_changes
        Vb_new = self.Vb + dVdb

        try:
            cVb_new = np.linalg.cholesky(Vb_new)
        except np.linalg.LinAlgError:
            cVb_new = np.linalg.cholesky(Vb_new + 1e-10 * np.eye(Vb_new.shape[0]))

        # Draw new function samples from the innovated GP on the representer points
        f_new = np.dot(cVb_new, self.F.T)
        f_new = f_new[:, :, None]
        Mb_new = Mb_new[:, None, :]
        f_new = Mb_new + f_new
        # Return the fantasized pmin
        return self.calc_pmin(f_new)

    def dh_fun(self, x):
        if x.shape[0] > 1:
            raise ValueError("EntropyMC is only for single test points")
        # Compute the fantasized pmin if we would evaluate at x
        new_pmin = self.change_pmin_by_innovation(x)

        # Calculate the Kullback-Leibler divergence w.r.t. this pmin approximation
        H_old = np.sum(np.multiply(self.pmin, (self.logP + self.lmb)))
        H_new = np.sum(np.multiply(new_pmin, (np.log(new_pmin) + self.lmb)))

        # Return the expected information gain
        return np.array([[-H_new + H_old]])
