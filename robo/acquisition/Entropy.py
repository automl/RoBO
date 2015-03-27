# encoding=utf8
import sys, os
from scipy.stats import norm
import scipy
import numpy as np
import emcee
import copy
from robo.loss_functions import logLoss
from robo import BayesianOptimizationError
from robo.sampling import sample_from_measure
from robo.maximize import _scipy_optimizer_fkt_wrapper
from robo.acquisition.LogEI import LogEI
from robo.acquisition.UCB import UCB
from robo.acquisition.base import AcquisitionFunction 
sq2 = np.sqrt(2)
l2p = np.log(2) + np.log(np.pi)
eps = np.finfo(np.float32).eps
here = os.path.abspath(os.path.dirname(__file__))

class Entropy(AcquisitionFunction):
    r"""
    The Entropy Search acquisition function minimize a loss function math:`\mathcal{L}_{KL}` by maximizing its difference after predicting an evaluation at X.
      
      .. math::
    
        \text{EntropySearch}(X) := \mathcal{L}_{KL}(p_\text{min}, b) - \mathcal{L}_{KL}(p^X_\text{min}, b)
    
    Where :math:`\mathcal{L}_{KL}` defines the Kullback-Leibler divergence between the probability measure of the minimum and the uniform pdf (:math:`b`):
      
      .. math::
         
         \mathcal{L}_{KL}(p, b) :=  - \int p(x)\log\frac{p(x)}{b(x)}dx \\
         p_\text{min}(X) := \mathbb{P}(X = \arg\limits_{x}\min\limits_{x, f}f(x)) 
    
      
    :param model: A model should have following methods:
    
        - predict(X)
        - predict_variance(X1, X2)
    :param X_lower: Lower bounds for the search, its shape should be 1xD (D = dimension of search space)
    :type X_lower: np.ndarray (1,D)
    :param X_upper: Upper bounds for the search, its shape should be 1xD (D = dimension of search space)
    :type X_upper: np.ndarray (1,D)
    :param Nb: Number of representer points to define :math:`p_\text{min}` at.
    :type Nb: int
    :param sampling_acquisition: A function to be used in calculating the density that representer points are to be sampled from. It uses
    :type samping_acquisition: AcquisitionFunction
    :param sampling_acquisition_kw: Additional keyword parameters to be passed to sampling_acquisition, if they are required, e.g. :math:`\xi` parameter for LogEI.
    :type sampling_acquisition_kw: dict
    :param Np: Number of prediction points at X to calculate stochastic changes of the mean for the representer points 
    :type Np: int
    :param loss_function: The loss function to be used in the calculation of the entropy. If not specified it deafults to log loss (cf. loss_functions module).
    """
    long_name = "Information gain over p_min(x)" 
    def __init__(self, model, X_lower, X_upper, Nb=10, sampling_acquisition=None, sampling_acquisition_kw={"par":0.0}, Np=400, loss_function=None, **kwargs):
        self.model = model
        self.Nb = Nb 
        self.X_lower = np.array(X_lower)
        self.X_upper = np.array(X_upper)
        self.D = self.X_lower.shape[0]
        self.BestGuesses = np.zeros((0, X_lower.shape[0]))
        if sampling_acquisition is None:
            sampling_acquisition = LogEI
        self.sampling_acquisition = sampling_acquisition(model, self.X_lower, self.X_upper, **sampling_acquisition_kw)
        if loss_function is None:
            loss_function = logLoss
        self.loss_function = loss_function
        self.Np = Np
        
    
    def _get_most_probable_minimum(self):
        acq = UCB(self.model, self.X_lower, self.X_upper, 0.0)
        sc_fun = _scipy_optimizer_fkt_wrapper(acq, derivative=False)
        minima = []
        for i in range(self.BestGuesses.shape[0]):
            xx = self.BestGuesses[i]
            minima.append(scipy.optimize.minimize(
                   fun=sc_fun, x0=xx, jac=False, method='L-BFGS-B', constraints=None,
                   options={'ftol':np.spacing(1), 'maxiter':120}
                ))
            
        Xdh = np.array([res.fun for res in minima])
        Xend = np.array([res.x for res in minima])
        new_x = Xend[np.nanargmin(Xdh)]
        return np.array([new_x]) 

    def __call__(self, X,  derivative=False, **kwargs):
        """
        :param x: The point at which the function is to be evaluated. Its shape is (1,D), where D is the dimension of the search space.
        :type x: np.ndarray (1, D)
        :param derivative: Controls whether the derivative is calculated and returned.
        :type derivative: Boolean
        :return: The expected difference of the loss function at X and optionally its derivative.
        :rtype: np.ndarray(1, 1) or (np.ndarray(1, 1), np.ndarray(1, D)).
        :raises BayesianOptimizationError: if X.shape[0] > 1. Only single X can be evaluated.
        """
        if X.shape[0] > 1 :
            raise BayesianOptimizationError(BayesianOptimizationError.SINGLE_INPUT_ONLY, "Entropy is only for single X inputs")
        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            if derivative:
                f = 0
                df = np.zeros((1, X.shape[1]))
                return np.array([[f]]), np.array([df])
            else:
                return np.array([[0]])
        return self.dh_fun(X, invertsign=True, derivative=derivative)
    
    def sampling_acquisition_wrapper(self, x):
        return  self.sampling_acquisition(np.array([x]))[0]
    
    def update_representer_points(self):
        self.sampling_acquisition.update(self.model)
        restarts = np.zeros((self.Nb, self.D))
        restarts[0:self.Nb, ] = self.X_lower + (self.X_upper - self.X_lower) * np.random.uniform(size=(self.Nb, self.D))
        sampler = emcee.EnsembleSampler(self.Nb, self.D, self.sampling_acquisition_wrapper)
        self.zb, self.lmb, _ = sampler.run_mcmc(restarts, 20)
        if len(self.zb.shape) == 1:
            self.zb = self.zb[:, None]
        if len(self.lmb.shape) == 1:
            self.lmb = self.lmb[:, None]
            
    def update_buest_guesses(self):
        if self.BestGuesses.shape[0] == 0:
            cmin = np.inf
        else:
            m = self.BestGuesses - self.zb[np.argmax(self.logP + self.lmb)]
            sqm = m * m
            c = np.sqrt(np.sum(sqm, axis=1))
            cmin = c.min() 
        if cmin < 0.25:
            self.BestGuesses[c.argmin()] = self.zb[np.argmax(self.logP + self.lmb)]
        else:
            self.BestGuesses = np.append(self.BestGuesses, np.array([self.zb[np.argmax(self.logP + self.lmb)]]), axis=0)
            
    def update(self, model):
        self.model = model
        self.update_representer_points()
        mu, var = self.model.predict(np.array(self.zb), full_cov=True)
        self.logP, self.dlogPdMu, self.dlogPdSigma, self.dlogPdMudMu = self._joint_min(mu, var, with_derivatives=True)
        self.W = np.random.randn(1, self.Nb)
        self.logP = np.reshape(self.logP, (self.logP.shape[0], 1))
        self.update_buest_guesses()
        
    def _dh_fun(self, x):
        # Number of belief locations:
        N = self.logP.size

        # Evaluate innovation
        Lx, _ = self._gp_innovation_local(x)
        # Innovation function for mean:
        dMdx = Lx
        # Innovation function for covariance:
        dVdx = -Lx.dot(Lx.T)
        # The transpose operator is there to make the array indexing equivalent to matlab's
        dVdx = dVdx[np.triu(np.ones((N, N))).T.astype(bool), np.newaxis]

        dMM = dMdx.dot(dMdx.T)
        trterm = np.sum(np.sum(
            np.multiply(self.dlogPdMudMu, np.reshape(dMM, (1, dMM.shape[0], dMM.shape[1]))),
            2), 1)[:, np.newaxis]

        # add a second dimension to the arrays if necessary:
        logP = np.reshape(self.logP, (self.logP.shape[0], 1))

        # Deterministic part of change:
        detchange = self.dlogPdSigma.dot(dVdx) + 0.5 * trterm
        # Stochastic part of change:
        stochange = (self.dlogPdMu.dot(dMdx)).dot(self.W)
        # Predicted new logP:
        
        lPred = np.add(logP + detchange, stochange)
        #
        _maxLPred = np.amax(lPred, axis=0)
        s = _maxLPred + np.log(np.sum(np.exp(lPred - _maxLPred), axis=0))
        lselP = _maxLPred if np.any(np.isinf(s)) else s
        #
        # lselP = np.log(np.sum(np.exp(lPred), 0))[np.newaxis,:]
        # Normalise:
        lPred = np.subtract(lPred, lselP)
        
        dHp = self.loss_function(logP, self.lmb, lPred, self.zb)
        dH = np.mean(dHp)
        return dH
        
    def dh_fun(self, x, invertsign=True, derivative=False):
        
        if not (np.all(np.isfinite(self.lmb))):
            print self.zb[np.where(np.isinf(self.lmb))], self.lmb[np.where(np.isinf(self.lmb))]
            raise Exception("lmb should not be infinite. This is not allowed to be sampled")
        
        D = x.shape[1]
        # If x is a vector, convert it to a matrix (some functions are sensitive to this distinction)
        if len(x.shape) == 1:
            x = x[np.newaxis]
            
        if np.any(x < self.X_lower) or np.any(x > self.X_upper):
            dH = np.spacing(1)
            ddHdx = np.zeros((x.shape[1], 1))
            return np.array([[dH]]), np.array([[ddHdx]])
        
        
        dH = self._dh_fun(x)

        if invertsign:
            dH = -dH
        if not np.isreal(dH):
            raise Exception("dH is not real")
        # Numerical derivative, renormalisation makes analytical derivatives unstable.
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
                if invertsign:
                    ddHdx = -ddHdx
            # endfor
            if len(ddHdx.shape) == 3:
                return_df = ddHdx
            else:
                return_df = np.array([ddHdx])
            return np.array([[dH]]), return_df
        return np.array([[dH]])

    def _gp_innovation_local(self, x):
        
        if x.shape[0] > 1:
            raise BayesianOptimizationError(BayesianOptimizationError.SINGLE_INPUT_ONLY, "single inputs please")

        m, v = self.model.predict(x)
        s = np.sqrt(v)
        v_projected = self.model.predict_variance(x, self.zb)
        Lx = v_projected / s;
        dLxdx = None
        return Lx, dLxdx
    
    def _joint_min(self, mu, var, with_derivatives=False, **kwargs):

        logP = np.zeros(mu.shape)
        D = mu.shape[0]
        if with_derivatives:
            dlogPdMu = np.zeros((D, D));
            dlogPdSigma = np.zeros((D, 0.5 * D * (D + 1)));
            dlogPdMudMu = np.zeros((D, D, D));
        for i in xrange(mu.shape[0]):
            
            # logP[k] ) self._min_faktor(mu, var, 0)
            a = self._min_faktor(mu, var, i)
            
            logP[i] = a.next()            
            if with_derivatives:
                dlogPdMu[i, :] = a.next().T
                dlogPdMudMu[i, :, :] = a.next()
                dlogPdSigma[i, :] = a.next().T
            
        logP[np.isinf(logP)] = -500;    
        # re-normalize at the end, to smooth out numerical imbalances:
        logPold = logP
        Z = np.sum(np.exp(logPold));
        _maxLogP = np.max(logP)
        s = _maxLogP + np.log(np.sum(np.exp(logP - _maxLogP)))
        s = _maxLogP if np.isinf(s) else s
        
        logP = logP - s;
        if not with_derivatives:
            return logP
        
        dlogPdMuold = dlogPdMu
        dlogPdSigmaold = dlogPdSigma
        dlogPdMudMuold = dlogPdMudMu;
        # adjust derivatives, too. This is a bit tedious.
        Zm = sum(np.rot90((np.exp(logPold) * np.rot90(dlogPdMuold, 1)), 3)) / Z
        Zs = sum(np.rot90((np.exp(logPold) * np.rot90(dlogPdSigmaold, 1)), 3)) / Z 
        
        
        dlogPdMu = dlogPdMuold - Zm
        dlogPdSigma = dlogPdSigmaold - Zs
    
        ff = np.einsum('ki,kj->kij', dlogPdMuold, dlogPdMuold)
        gg = np.einsum('kij,k->ij', dlogPdMudMuold + ff, np.exp(logPold)) / Z;
        Zij = Zm.T * Zm;
        adds = np.reshape(-gg + Zij, (1, D, D));
        dlogPdMudMu = dlogPdMudMuold + adds
        return logP, dlogPdMu, dlogPdSigma, dlogPdMudMu
            
    def _min_faktor(self, Mu, Sigma, k, gamma=1):

        D = Mu.shape[0]
        logS = np.zeros((D - 1,))
        # mean time first moment
        MP = np.zeros((D - 1,))
        
        # precision, second moment 
        P = np.zeros((D - 1,))
        
        M = np.copy(Mu)
        V = np.copy(Sigma)
        b = False
        for count in xrange(50):
            diff = 0
            for i in range(D - 1):
                l = i if  i < k else i + 1
                try:
                    M, V, P[i], MP[i], logS[i], d = self._lt_factor(k, l, M, V, MP[i], P[i], gamma)
                except Exception, e:
                    raise
                
                if np.isnan(d): 
                    break;
                diff += np.abs(d)
            if np.isnan(d): 
                    break;
            if np.abs(diff) < 0.001:
                b = True
                break;
        if np.isnan(d): 
            logZ = -np.Infinity;
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
            mpm = sum(mpm);
            
            s = sum(logS);
            IRSR = (np.eye(D - 1) + np.dot(np.dot(R.T , Sigma), R));
            rSr = np.dot(np.dot(r.T, Sigma) , r);
            A = np.dot(R, np.linalg.solve(IRSR, R.T)) 
            
            A = 0.5 * (A.T + A)  # ensure symmetry.
            b = (Mu + np.dot(Sigma, r));
            Ab = np.dot(A, b);
            try:
                cIRSR =  np.linalg.cholesky(IRSR)   
            except np.linalg.LinAlgError:
                try:
                    cIRSR = np.linalg.cholesky(IRSR + 1e-10 * np.eye(IRSR.shape[0]))
                except np.linalg.LinAlgError:
                    cIRSR = np.linalg.cholesky(IRSR + 1e-6 * np.eye(IRSR.shape[0]))
            dts = 2 * np.sum(np.log(np.diagonal(cIRSR)));
            logZ = 0.5 * (rSr - np.dot(b.T, Ab) - dts) + np.dot(Mu.T, r) + s - 0.5 * mpm;
            yield logZ
            btA = np.dot(b.T, A)
            
            dlogZdMu = r - Ab
            yield dlogZdMu
            dlogZdMudMu = -A
            yield dlogZdMudMu
            dlogZdSigma = -A - 2 * np.outer(r, Ab.T) + np.outer(r, r.T) + np.outer(btA.T, Ab.T);
            _dlogZdSigma = np.zeros_like(dlogZdSigma)
            np.fill_diagonal(_dlogZdSigma, np.diagonal(dlogZdSigma))
            dlogZdSigma = 0.5 * (dlogZdSigma + dlogZdSigma.T - _dlogZdSigma)
            dlogZdSigma = np.rot90(dlogZdSigma, k=2)[np.triu_indices(D)][::-1];
            yield dlogZdSigma
            
    def _lt_factor(self, s, l, M, V, mp, p, gamma):

        cVc = (V[l, l] - 2 * V[s, l] + V[s, s]) / 2.0
        Vc = (V[:, l] - V [:, s]) / sq2
        cM = (M[l] - M[s]) / sq2
        cVnic = np.max([cVc / (1 - p * cVc), 0])
        cmni = cM + cVnic * (p * cM - mp)
        z = cmni / np.sqrt(cVnic);
        if np.isnan(z):
            z = -np.inf
        e, lP, exit_flag = self._log_relative_gauss(z)
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
            dp = np.max([-p + eps, gamma * (pnew - p)])  # at worst, remove message
            dmp = np.max([-mp + eps, gamma * (mpnew - mp)])
            d = np.max([dmp, dp])  # for convergence measures
        
            pnew = p + dp;
            mpnew = mp + dmp;
            # project out to marginal
            Vnew = V - dp / (1 + dp * cVc) * np.outer(Vc, Vc)
            
            Mnew = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc
            if np.any(np.isnan(Vnew)): raise Exception("an error occurs while running expectation propagation in entropy search. Resulting variance contains NaN")
            # % there is a problem here, when z is very large
            logS = lP - 0.5 * (np.log(beta) - np.log(pnew) - np.log(cVnic)) + (alpha * alpha) / (2 * beta) * cVnic
             
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
            d = max([dmp, dp]);  # for convergence measures
            # project out to marginal
            Vnew = V - dp / (1 + dp * cVc) * (np.outer(Vc, Vc))
            Mnew = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc;
            logS = 0;
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
        
    def plot(self, fig, minx, maxx, plot_attr={"color":"red"}, resolution=1000):

        n = len(fig.axes)
        for i in range(n):
            fig.axes[i].change_geometry(n + 2, 1, i + 1) 
        ax = fig.add_subplot(n + 1, 1, n + 1)
        bar_ax = fig.add_subplot(n + 2, 1, n + 2)
        plotting_range = np.linspace(minx, maxx, num=resolution)
        acq_v = np.array([ self(np.array([x]), derivative=True)[0][0] for x in plotting_range[:, np.newaxis] ])
        ax.plot(plotting_range, acq_v, **plot_attr)
       
        zb = self.zb
        pmin = np.exp(self.logP)
        ax.set_xlim(minx, maxx)
        bar_ax.bar(zb, pmin, width=(maxx - minx) / (2 * zb.shape[0]), color="yellow")
        bar_ax.set_xlim(minx, maxx)
        bar_ax.set_ylim(0.0, pmin.max())
        other_acq_ax = self.sampling_acquisition.plot(fig, minx, maxx, plot_attr={"color":"orange"})  # , logscale=True)
        other_acq_ax.set_xlim(minx, maxx)
        ax.set_title(str(self))
        return ax
