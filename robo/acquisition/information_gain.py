# encoding=utf8

import sys
import logging

import scipy
import numpy as np
import emcee


from robo.acquisition.log_ei import LogEI
from robo.acquisition.base import AcquisitionFunction


sq2 = np.sqrt(2)
l2p = np.log(2) + np.log(np.pi)
eps = np.finfo(np.float32).eps

logger = logging.getLogger(__name__)


class InformationGain(AcquisitionFunction):

    def __init__(self, model, X_lower, X_upper,
            Nb=10, Np=400, sampling_acquisition=None,
            sampling_acquisition_kw={"par": 0.0},
            **kwargs):

        """
        The Information Gain acquisition function for Entropy Search [1].
        In a nutshell entropy search approximates the
        distribution pmin of the global minimum and tries to decrease its
        entropy. See Hennig and Schuler[1] for a more detailed view.

        [1] Hennig and C. J. Schuler
            Entropy search for information-efficient global optimization
            Journal of Machine Learning Research, 13, 2012

        Parameters
        ----------
        model: Model object
            A model that implements at least
                 - predict(X)
                 - predict_variances(X1, X2)
            If you want to calculate derivatives than it should also support
                 - predictive_gradients(X)
        X_lower: np.ndarray (D)
            Lower bounds of the input space
        X_upper: np.ndarray (D)
            Upper bounds of the input space
        Nb: int
            Number of representer points to define pmin.
        Np: int
            Number of hallucinated values to compute the innovations
            at the representer points
        sampling_acquisition: function
            Proposal measurement from which the representer points will
            be samples
        sampling_acquisition_kw: dist
            Additional keyword parameters that are passed to the
            acquisition function
        """

        self.Nb = Nb
        super(InformationGain, self).__init__(model, X_lower, X_upper)
        self.D = self.X_lower.shape[0]
        self.sn2 = None

        if sampling_acquisition is None:
            sampling_acquisition = LogEI
        self.sampling_acquisition = sampling_acquisition(
            model, self.X_lower, self.X_upper, **sampling_acquisition_kw)

        self.Np = Np

    def loss_function(self, logP, lmb, lPred, *args):

        H = -np.sum(np.multiply(np.exp(logP), (logP + lmb)))  # current entropy
        dHp = - np.sum(np.multiply(np.exp(lPred),
                                   np.add(lPred, lmb)), axis=0) - H
        return np.array([dHp])

    def compute(self, X, derivative=False, **kwargs):

        if X.shape[0] > 1:
            raise ValueError("Entropy is only for single test points")
        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            if derivative:
                f = 0
                df = np.zeros((1, X.shape[1]))
                return np.array([[f]]), np.array([df])
            else:
                return np.array([[0]])

        if derivative:
            acq, grad = self.dh_fun(X, derivative=True)

            if np.any(np.isnan(acq)) or np.any(acq == np.inf):
                return -sys.float_info.max
            return acq, grad
        else:
            acq = self.dh_fun(X, derivative=False)

            if np.any(np.isnan(acq)) or np.any(acq == np.inf):
                return -sys.float_info.max
            return acq

    def sampling_acquisition_wrapper(self, x):
        if np.any(x < self.X_lower) or np.any(x > self.X_upper):
            return -np.inf
        return self.sampling_acquisition(np.array([x]))[0]

    def sample_representer_points(self):
        self.sampling_acquisition.update(self.model)
        restarts = np.zeros((self.Nb, self.D))
        restarts[0:self.Nb, ] = self.X_lower + (self.X_upper - self.X_lower) \
                    * np.random.uniform(size=(self.Nb, self.D))
        sampler = emcee.EnsembleSampler(
            self.Nb, self.D, self.sampling_acquisition_wrapper)
        # zb are the representer points and lmb are their log EI values
        self.zb, self.lmb, _ = sampler.run_mcmc(restarts, 20)
        if len(self.zb.shape) == 1:
            self.zb = self.zb[:, None]
        if len(self.lmb.shape) == 1:
            self.lmb = self.lmb[:, None]

    def update(self, model):
        self.model = model

        self.sn2 = self.model.get_noise()
        self.sample_representer_points()
        mu, var = self.model.predict(np.array(self.zb), full_cov=True)

        self.logP, self.dlogPdMu, self.dlogPdSigma, self.dlogPdMudMu = \
                        self._joint_min(mu, var, with_derivatives=True)

        self.W = scipy.stats.norm.ppf(np.linspace(1. / (self.Np + 1),
                                    1 - 1. / (self.Np + 1),
                                    self.Np))[np.newaxis, :]

        self.logP = np.reshape(self.logP, (self.logP.shape[0], 1))

    def _dh_fun(self, x):
        # Number of belief locations:
        N = self.logP.size

        # Evaluate innovation
        dMdx, dVdx = self.innovations(x, self.zb)
        # The transpose operator is there to make the array indexing equivalent
        # to matlab's
        dVdx = dVdx[np.triu(np.ones((N, N))).T.astype(bool), np.newaxis]

        dMM = dMdx.dot(dMdx.T)
        trterm = np.sum(np.sum(np.multiply(self.dlogPdMudMu, np.reshape(
                        dMM, (1, dMM.shape[0], dMM.shape[1]))), 2), 1)[
            :, np.newaxis]

        # add a second dimension to the arrays if necessary:
        logP = np.reshape(self.logP, (self.logP.shape[0], 1))

        # Deterministic part of change:
        detchange = self.dlogPdSigma.dot(dVdx) + 0.5 * trterm
        # Stochastic part of change:
        stochange = (self.dlogPdMu.dot(dMdx)).dot(self.W)
        # Predicted new logP:
        lPred = np.add(logP + detchange, stochange)
        _maxLPred = np.amax(lPred, axis=0)
        s = _maxLPred + np.log(np.sum(np.exp(lPred - _maxLPred), axis=0))
        lselP = _maxLPred if np.any(np.isinf(s)) else s

        # Normalise:
        lPred = np.subtract(lPred, lselP)

        # We maximize the information gain
        dHp = -self.loss_function(logP, self.lmb, lPred, self.zb)
        dH = np.mean(dHp)
        return dH

    def dh_fun(self, x, derivative=False):

        if not (np.all(np.isfinite(self.lmb))):
            logger.debug(
                self.zb[
                    np.where(
                        np.isinf(
                            self.lmb))], self.lmb[
                    np.where(
                        np.isinf(
                            self.lmb))])
            raise Exception(
                "lmb should not be infinite.")

        D = x.shape[1]
        # If x is a vector, convert it to a matrix (some functions are
        # sensitive to this distinction)
        if len(x.shape) == 1:
            x = x[np.newaxis]

        if np.any(x < self.X_lower) or np.any(x > self.X_upper):
            dH = np.spacing(1)
            ddHdx = np.zeros((x.shape[1], 1))
            return np.array([[dH]]), np.array([[ddHdx]])

        dH = self._dh_fun(x)

        if not np.isreal(dH):
            raise Exception("dH is not real")
        # Numerical derivative, renormalisation makes analytical derivatives
        # unstable.
        e = 1.0e-5
        if derivative:
            ddHdx = np.zeros((1, D))
            for d in range(D):
                # ## First part:
                y = np.array(x)
                y[0, d] += e
                dHy1 = self._dh_fun(y)
                # ## Second part:
                y = np.array(x)
                y[0, d] -= e
                dHy2 = self._dh_fun(y)

                ddHdx[0, d] = np.divide((dHy1 - dHy2), 2 * e)
                ddHdx = -ddHdx
            # endfor
            if len(ddHdx.shape) == 3:
                return_df = ddHdx
            else:
                return_df = np.array([ddHdx])
            return np.array([[dH]]), return_df
        return np.array([[dH]])

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

        # Compute the deterministic innovation for the variance
        dv_rep = -norm_cov.dot(sigma_x_rep.T)

        return dm_rep, dv_rep

    def _joint_min(self, mu, covar, with_derivatives=False, **kwargs):
        mu = mu[:, 0]
        logP = np.zeros(mu.shape)
        D = mu.shape[0]
        if with_derivatives:
            dlogPdMu = np.zeros((D, D))
            dlogPdSigma = np.zeros((D, 0.5 * D * (D + 1)))
            dlogPdMudMu = np.zeros((D, D, D))
        # Loop over representer points to compute p_min(x_i)
        for i in range(mu.shape[0]):

            # logP[k] ) self._min_faktor(mu, var, 0)
            a = self._min_faktor(mu, covar, i)

            logP[i] = next(a)
            if with_derivatives:
                dlogPdMu[i, :] = a.next().T
                dlogPdMudMu[i, :, :] = next(a)
                dlogPdSigma[i, :] = a.next().T

        logP[np.isinf(logP)] = -500
        # re-normalize at the end, to smooth out numerical imbalances:
        logPold = logP
        Z = np.sum(np.exp(logPold))
        _maxLogP = np.max(logP)
        s = _maxLogP + np.log(np.sum(np.exp(logP - _maxLogP)))
        s = _maxLogP if np.isinf(s) else s

        logP = logP - s
        if not with_derivatives:
            return logP

        dlogPdMuold = dlogPdMu
        dlogPdSigmaold = dlogPdSigma
        dlogPdMudMuold = dlogPdMudMu
        # adjust derivatives, too. This is a bit tedious.
        Zm = sum(np.rot90((np.exp(logPold) * np.rot90(dlogPdMuold, 1)), 3)) / Z
        Zs = sum(np.rot90((np.exp(logPold) * np.rot90(dlogPdSigmaold, 1)), 3)) / Z

        dlogPdMu = dlogPdMuold - Zm
        dlogPdSigma = dlogPdSigmaold - Zs

        ff = np.einsum('ki,kj->kij', dlogPdMuold, dlogPdMuold)
        gg = np.einsum('kij,k->ij', dlogPdMudMuold + ff, np.exp(logPold)) / Z
        Zij = Zm.T * Zm
        adds = np.reshape(-gg + Zij, (1, D, D))
        dlogPdMudMu = dlogPdMudMuold + adds
        return logP, dlogPdMu, dlogPdSigma, dlogPdMudMu

    # EP
    def _min_faktor(self, Mu, Sigma, k, gamma=1):
        # k index that defines the representer points
        # where we compute the cavity
        D = Mu.shape[0]
        logS = np.zeros((D - 1,))
        # mean time first moment (nu)
        MP = np.zeros((D - 1,))

        # precision, second moment (Tau)
        P = np.zeros((D - 1,))
        # the mean and covariance of q0
        M = np.copy(Mu)
        V = np.copy(Sigma)
        b = False
        for count in range(50):
            diff = 0
            for i in range(D - 1):
                # Take each dimension where i is not equal to k
                l = i if i < k else i + 1
                try:
                    M, V, P[i], MP[i], logS[i], d = self._lt_factor(
                        k, l, M, V, MP[i], P[i], gamma)
                except Exception as e:
                    raise

                if np.isnan(d):
                    break
                diff += np.abs(d)
            if np.isnan(d):
                break
            if np.abs(diff) < 0.001:
                b = True
                break
        if np.isnan(d):
            logZ = -np.Infinity
            yield logZ
            dlogZdMu = np.zeros((D, 1))
            yield dlogZdMu

            dlogZdMudMu = np.zeros((D, D))
            yield dlogZdMudMu
            dlogZdSigma = np.zeros((0.5 * (D * (D + 1)), 1))
            yield dlogZdSigma
            mvmin = [Mu[k], Sigma[k, k]]
            yield mvmin
        else:
            # evaluate log Z:
            C = np.eye(D) / sq2
            C[k, :] = -1 / sq2
            C = np.delete(C, k, 1)

            R = np.sqrt(P.T) * C
            r = np.sum(MP.T * C, 1)
            mp_not_zero = np.where(MP != 0)
            mpm = MP[mp_not_zero] * MP[mp_not_zero] / P[mp_not_zero]
            mpm = sum(mpm)

            s = sum(logS)
            IRSR = (np.eye(D - 1) + np.dot(np.dot(R.T, Sigma), R))
            rSr = np.dot(np.dot(r.T, Sigma), r)
            A = np.dot(R, np.linalg.solve(IRSR, R.T))

            A = 0.5 * (A.T + A)  # ensure symmetry.
            b = (Mu + np.dot(Sigma, r))
            Ab = np.dot(A, b)
            noise = 0
            while(True):
                try:
                    cIRSR = np.linalg.cholesky(
                        IRSR + noise * np.eye(IRSR.shape[0]))
                    break
                except np.linalg.LinAlgError:
                    if noise == 0:
                        noise = 1e-10
                    else:
                        noise *= 10
                    logger.error(
                        "Cholesky decomposition failed."\
                        "Add %f noise on the diagonal." %
                        noise)
            dts = 2 * np.sum(np.log(np.diagonal(cIRSR)))
            logZ = 0.5 * (rSr - np.dot(b.T, Ab) - dts) + \
                np.dot(Mu.T, r) + s - 0.5 * mpm
            yield logZ
            btA = np.dot(b.T, A)

            dlogZdMu = r - Ab
            yield dlogZdMu
            dlogZdMudMu = -A
            yield dlogZdMudMu
            dlogZdSigma = -A - 2 * \
                np.outer(r, Ab.T) + np.outer(r, r.T) + np.outer(btA.T, Ab.T)
            _dlogZdSigma = np.zeros_like(dlogZdSigma)
            np.fill_diagonal(_dlogZdSigma, np.diagonal(dlogZdSigma))
            dlogZdSigma = 0.5 * (dlogZdSigma + dlogZdSigma.T - _dlogZdSigma)
            dlogZdSigma = np.rot90(dlogZdSigma, k=2)[np.triu_indices(D)][::-1]
            yield dlogZdSigma

    def _lt_factor(self, k, i, M, V, mp, p, gamma):
        # k is the current x_i (aka the representer where we evaluate pmin)
        # i is an index of a representer point
        # p is equal to tau
        # mp is equal to nu
        # ci = is zeros everywhere except:
        #     c[k] = -1 / sqrt(2) , c[i] = 1 / sqrt(2)
        # Cavity parameters (Equation 51)
        cVc = (V[i, i] - 2 * V[k, i] + V[k, k]) / 2.0  # ciT * Sigma * ci
        Vc = (V[:, i] - V[:, k]) / sq2  # Sigma * ci
        cM = (M[i] - M[k]) / sq2  # mu * ci
        cVnic = np.max([cVc / (1 - p * cVc), 0])  # sigma_/i ** 2
        cmni = cM + cVnic * (p * cM - mp)  # mu_/i
        z = cmni / np.sqrt(cVnic)
        if np.isnan(z):
            z = -np.inf
        e, lP, exit_flag = self._log_relative_gauss(z)
        # Projection step
        if exit_flag == 0:
            alpha = e / np.sqrt(cVnic)
            # beta  = alpha * (alpha + cmni / cVnic);
            # r     = beta * cVnic / (1 - cVnic * beta);
            beta = alpha * (alpha * cVnic + cmni)
            r = beta / (1 - beta)
            # new message
            pnew = r / cVnic
            mpnew = r * (alpha + cmni / cVnic) + alpha

            # update terms
            # at worst, remove message
            # delta tau = tau_new - tau_old
            dp = np.max([-p + eps, gamma * (pnew - p)])
            # delta nu = nu_new - nu_old
            dmp = np.max([-mp + eps, gamma * (mpnew - mp)])
            d = np.max([dmp, dp])  # for convergence measures

            pnew = p + dp
            mpnew = mp + dmp
            # project out to marginal
            Vnew = V - dp / (1 + dp * cVc) * np.outer(Vc, Vc)  # Equation 57

            Mnew = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc  # Equation 58
            if np.any(np.isnan(Vnew)):
                raise Exception(
                    "an error occurs while running expectation"\
                    "propagation in entropy search."\
                    "Resulting variance contains NaN")
            # % there is a problem here, when z is very large
            logS = lP - 0.5 * (np.log(beta) - np.log(pnew) -
                        np.log(cVnic)) + (alpha * alpha) / (2 * beta) * cVnic

        elif exit_flag == -1:
            d = np.NAN
            Mnew = 0
            Vnew = 0
            pnew = 0
            mpnew = 0
            logS = -np.Infinity
        elif exit_flag == 1:
            d = 0
            # remove message from marginal:
            # new message
            pnew = 0
            mpnew = 0
            # update terms
            dp = -p  # at worst, remove message
            dmp = -mp
            d = max([dmp, dp])  # for convergence measures
            # project out to marginal
            Vnew = V - dp / (1 + dp * cVc) * (np.outer(Vc, Vc))
            Mnew = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc
            logS = 0
        return Mnew, Vnew, pnew, mpnew, logS, d

    def _log_relative_gauss(self, z):
        if z < -6:
            return 1, -1.0e12, -1
        if z > 6:
            return 0, 0, 1
        else:
            logphi = -0.5 * (z * z + l2p)
            logPhi = np.log(.5 * scipy.special.erfc(-z / sq2))
            e = np.exp(logphi - logPhi)
            return e, logPhi, 0
