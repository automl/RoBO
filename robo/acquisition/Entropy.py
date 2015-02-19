import sys, os
from scipy.stats import norm
import scipy
import numpy as np
from robo.loss_functions import logLoss
from robo import BayesianOptimizationError
from robo.sampling import sample_from_measure
from robo.acquisition.EI import EI
sq2 = np.sqrt(2)
l2p = np.log(2) + np.log(np.pi)
eps = np.finfo(np.float32).eps

here = os.path.abspath(os.path.dirname(__file__))
class Entropy(object):
    
    def __init__(self, model, X_lower, X_upper, Nb = 100, sampling_acquisition = None, sampling_acquisition_kw = {"par":2.4}, T=200, loss_function=None, **kwargs):
        self.model = model
        self.Nb = Nb 
        self.X_lower = np.array(X_lower)
        self.X_upper = np.array(X_upper)
        self.BestGuesses = np.zeros((0, X_lower.shape[0]))
        if sampling_acquisition is None:
            sampling_acquisition = EI
        self.sampling_acquisition = sampling_acquisition(model, self.X_lower, self.X_upper, **sampling_acquisition_kw)
        if loss_function is None:
            loss_function = logLoss
        self.loss_function = loss_function        
        self.T = T
    
    def _get_most_probable_minimum(self):
        mi = np.argmax(self.logP)
        xx = self.zb[mi,np.newaxis]
        return xx
        
    def __call__(self, X, Z=None, derivative=False, **kwargs):
        return self.dh_fun(X, invertsign=True, derivative=derivative)

    def update(self, model):
        self.model = model
        self.sampling_acquisition.update(model)
        self.zb, self.lmb = sample_from_measure(self.model, self.X_lower, self.X_upper, self.Nb, self.BestGuesses, self.sampling_acquisition)
        mu, var = self.model.predict(np.array(self.zb), full_cov=True)
        self.logP,self.dlogPdMu,self.dlogPdSigma,self.dlogPdMudMu = self._joint_min(mu, var, with_derivatives=True)
        self.current_entropy = - np.sum (np.exp(self.logP) * (self.logP+self.lmb) )
        self.W = np.random.randn(1, self.T)
        self.K = self.model.K
        self.cK = self.model.cK.T
        self.kbX = self.model.kernel.K(self.zb,self.model.X)
        self.logP = np.reshape(self.logP, (self.logP.shape[0], 1))



    def _dh_fun(self, x):
            # Number of belief locations:
        N = self.logP.size
        
        #T = self.T
        
        # Evaluate innovation
        Lx, _ = self._gp_innovation_local(x)
        # Innovation function for mean:
        dMdx = Lx
        # Innovation function for covariance:
        dVdx = -Lx.dot(Lx.T)
        # The transpose operator is there to make the array indexing equivalent to matlab's
        dVdx = dVdx[np.triu(np.ones((N,N))).T.astype(bool), np.newaxis]

        dMM = dMdx.dot(dMdx.T)
        trterm = np.sum(np.sum(
            np.multiply(self.dlogPdMudMu, np.reshape(dMM, (1, dMM.shape[0],dMM.shape[1]))),
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
        #lselP = np.log(np.sum(np.exp(lPred), 0))[np.newaxis,:]
        # Normalise:
        lPred = np.subtract(lPred, lselP)
        
        dHp = self.loss_function(logP, self.lmb, lPred, self.zb)
        dH = np.mean(dHp)
        return dH
        
    def dh_fun(self, x, invertsign = True, derivative=False):
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
            return dH, ddHdx
        
        if x.shape[0] > 1:
            raise BayesianOptimizationError(BayesianOptimizationError.SINGLE_INPUT_ONLY, "dHdx_local is only for single x inputs")
        
        dH = self._dh_fun(x)

        if invertsign:
            dH = - dH
        if not np.isreal(dH):
            raise Exception("dH is not real")
        # Numerical derivative, renormalisation makes analytical derivatives unstable.
        e = 1.0e-5
        if derivative:
            ddHdx = np.zeros((D,1))
            for d in range(D):
                ### First part:
                y = np.array(x)
                y[d] += e
                dHy1 = self._dh_fun(y)
                ### Second part:
                y = np.array(x)
                y[d] -= e
                dHy2 = self._dh_fun(y)
                
                ddHdx[d] = np.divide((dHy1 - dHy2), 2*e)
                if invertsign:
                    ddHdx = -ddHdx
            # endfor
            return dH, ddHdx
        return dH

    def _gp_innovation_local(self, x):
        zb = self.zb
        K = self.K
        cK = self.cK
        kbX = self.kbX
        if x.shape[0] > 1:
            raise BayesianOptimizationError(BayesianOptimizationError.SINGLE_INPUT_ONLY, "single inputs please")

        if self.model.X.shape[0] == 0:
            # kernel values
            kbx = self.model.kernel.K(zb,x)
            kXx = self.model.kernel.K(self.model.X, x)
            kxx = self.model.likelihood.variance + self.model.kernel.K(x, x)
            
            #derivatives of kernel values
            dkxx = self.model.kernel.gradients_X(kxx, x)
            dkxX = -1* self.model.kernel.gradients_X(np.ones((self.model.X.shape[0], x.shape[0])),self.model.X, x)
            dkxb = -1* self.model.kernel.gradients_X(np.ones((zb.shape[0], x.shape[0])), zb, x)
            # terms of innovation
            a = kxx - np.dot(kXx.T, (np.linalg.solve(cK, np.linalg.solve(cK.T, kXx))))
            sloc = np.sqrt(a)
            proj = kbx - np.dot(kbX, np.linalg.solve(cK, np.linalg.solve(cK.T, kXx)))
            
            #terms of the innovation
            sloc   = np.sqrt(kxx)
            proj   = kbx
            
            dvloc  = dkxx
            dproj  = dkxb
            
            # innovation, and its derivative
            Lx     = proj / sloc;
            dLxdx  = dproj / sloc - 0.5 * proj * dvloc / (sloc**3);
            return Lx, dLxdx
        #m, kxx = self.model.predict(x)
        #s = np.sqrt(v)
        #dmdx, ds2dx = self.model.m.predictive_gradients(x)
        #dsdx = ds2dx / (2*s)
            
        kbx = self.model.kernel.K(zb,x)
        kXx = self.model.kernel.K(self.model.X, x)
        #kxx = self.model.likelihood.variance +
        kxx = self.model.kernel.K(x, x) #TODO Joel
        # derivatives of kernel values 
        dkxx = self.model.kernel.gradients_X(np.ones((kxx.shape[0], x.shape[0])), kxx, x)
        dkxX = -1* self.model.kernel.gradients_X(np.ones((self.model.X.shape[0], x.shape[0])),self.model.X, x)
        dkxb = -1* self.model.kernel.gradients_X(np.ones((zb.shape[0], x.shape[0])), zb, x)
        
        matlab_matrices =scipy.io.loadmat(here+'/../../../entropie_search/EntropySearch/pqfile.mat')
        # terms of innovation
        a = kxx - np.dot(kXx.T, (np.linalg.solve(cK, np.linalg.solve(cK.T, kXx))))
        m, v = self.model.predict(x)
        sloc1 = np.sqrt(v)
        matlab_matrices['kxx'] -np.dot(matlab_matrices['kXx'].T, (np.linalg.solve(cK, np.linalg.solve(cK.T, matlab_matrices['kXx']))))
        #posterior, self._log_marginal_likelihood, self.grad_dict = self.model.m.inference_method.inference(self.model.m.kern, self.model.m.X, self.model.m.likelihood, self.model.m.Y_normalized, self.model.m.Y_metadata)
        sloc = np.sqrt(a)
        proj = kbx - np.dot(kbX, np.linalg.solve(cK, np.linalg.solve(cK.T, kXx)))
        
        dvloc  = (dkxx.T - 2 * np.dot(dkxX.T, np.linalg.solve(cK, np.linalg.solve(cK.T, kXx)))).T;
        dproj  = dkxb - np.dot(kbX, np.linalg.solve(cK, np.linalg.solve(cK.T, dkxX)));
        
        #innovation, and its derivative
        Lx     = proj / sloc;
        dLxdx  = dproj / sloc - 0.5 * proj * dvloc / (sloc**3);
        return Lx, dLxdx
    
    def _joint_min(self, mu, var, with_derivatives= False, **kwargs):
        logP = np.zeros(mu.shape)
        D = mu.shape[0]
        if with_derivatives:
            dlogPdMu    = np.zeros((D,D));
            dlogPdSigma = np.zeros((D,0.5 * D * (D+1)));
            dlogPdMudMu= np.zeros((D,D,D));
        for i in xrange(mu.shape[0]):
            
            #logP[k] ) self._min_faktor(mu, var, 0)
            a = self._min_faktor(mu, var, i)
            
            logP[i] = a.next()            
            if with_derivatives:
                dlogPdMu[i,:] = a.next().T
                dlogPdMudMu[i, :, :] = a.next()
                dlogPdSigma[i,:] = a.next().T
            
        logP[np.isinf(logP)] = -500;    
        #re-normalize at the end, to smooth out numerical imbalances:
        logPold        = logP
        Z     = np.sum(np.exp(logPold));
        _maxLogP = np.max(logP)
        s = _maxLogP + np.log(np.sum(np.exp(logP - _maxLogP)))
        s = _maxLogP if np.isinf(s) else s
        
        logP  = logP - s;
        if not with_derivatives:
            return logP
        
        dlogPdMuold    = dlogPdMu
        dlogPdSigmaold = dlogPdSigma
        dlogPdMudMuold = dlogPdMudMu;
        # adjust derivatives, too. This is a bit tedious.
        Zm    = sum(np.rot90((np.exp(logPold)*np.rot90(dlogPdMuold,1)),3)) / Z
        Zs    = sum(np.rot90((np.exp(logPold)*np.rot90(dlogPdSigmaold,1)),3)) / Z 
        
        
        dlogPdMu    = dlogPdMuold-Zm
        dlogPdSigma = dlogPdSigmaold - Zs
    
        ff = np.einsum('ki,kj->kij', dlogPdMuold, dlogPdMuold)
        gg   = np.einsum('kij,k->ij',dlogPdMudMuold+ff,np.exp(logPold)) / Z;
        Zij  = Zm.T * Zm;
        adds = np.reshape(-gg+Zij,(1,D,D));
        dlogPdMudMu = dlogPdMudMuold + adds
        return logP,dlogPdMu,dlogPdSigma,dlogPdMudMu
            
    def _min_faktor(self, Mu, Sigma, k, gamma = 1):
        D = Mu.shape[0]
        logS = np.zeros((D-1,))
        #mean time first moment
        MP = np.zeros((D-1,))
        
        #precision, second moment 
        P = np.zeros((D-1,))
        
        M = np.copy(Mu)
        V = np.copy(Sigma)
        b = False
        for count in xrange(50):
            diff = 0
            for i in range(D-1):
                l = i if  i < k else i+1
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
            logZ  = -np.Infinity;
            yield logZ
            dlogZdMu = np.zeros((D,1))
            yield dlogZdMu
            
            dlogZdMudMu = np.zeros((D,D))
            yield dlogZdMudMu
            dlogZdSigma = np.zeros((0.5*(D*(D+1)),1))
            yield dlogZdSigma
            mvmin = [Mu[k],Sigma[k,k]]
            yield mvmin
            dMdMu = np.zeros((1,D))
            yield dMdMu
            dMdSigma = np.zeros((1,0.5*(D*(D+1))))
            yield dMdSigma
            dVdSigma = np.zeros((1,0.5*(D*(D+1))))
            yield dVdSigma
        else:
            #evaluate log Z:
            C = np.eye(D) / sq2 
            C[k,:] = -1/sq2
            C = np.delete(C, k, 1)
            
            R       = np.sqrt(P.T) * C
            r       = np.sum(MP.T * C, 1)
            mp_not_zero = np.where(MP !=0)
            mpm = MP[mp_not_zero] * MP[mp_not_zero] / P[mp_not_zero]
            mpm     = sum(mpm);
            
            s       = sum(logS);
            IRSR    = (np.eye(D-1) + np.dot(np.dot(R.T , Sigma), R));
            rSr     = np.dot(np.dot(r.T, Sigma) , r);
            A =  np.dot(R,np.linalg.solve(IRSR,R.T)) 
            
            A       = 0.5 * (A.T + A) # ensure symmetry.
            b       = (Mu + np.dot(Sigma,r));
            Ab      = np.dot(A,b);
            dts     = 2 * np.sum(np.log(np.diagonal(np.linalg.cholesky(IRSR))));
            logZ    = 0.5 * (rSr - np.dot(b.T, Ab) - dts) + np.dot(Mu.T, r) + s - 0.5 * mpm;
            yield logZ
            btA = np.dot(b.T, A)
            
            dlogZdMu    = r - Ab
            yield dlogZdMu
            dlogZdMudMu = -A
            yield dlogZdMudMu
            dlogZdSigma = -A - 2*np.outer(r,Ab.T) + np.outer(r,r.T) + np.outer(btA.T,Ab.T);
            _dlogZdSigma = np.zeros_like(dlogZdSigma)
            np.fill_diagonal(_dlogZdSigma, np.diagonal(dlogZdSigma))
            dlogZdSigma = 0.5*(dlogZdSigma+dlogZdSigma.T-_dlogZdSigma)
            dlogZdSigma = np.rot90(dlogZdSigma, k=2)[np.triu_indices(D)][::-1];
            yield dlogZdSigma
            
    def _lt_factor(self, s, l, M, V, mp, p, gamma):

        cVc = (V[l,l] - 2*V[s,l] + V[s,s])/ 2.0
        Vc  = (V[:, l] - V [:, s]) / sq2
        cM =  (M[l] - M[s])/ sq2
        cVnic = np.max([cVc/(1-p * cVc), 0])
        cmni = cM + cVnic * (p * cM - mp)
        z     = cmni / np.sqrt(cVnic);
        e,lP,exit_flag = self._log_relative_gauss( z)
        if exit_flag == 0:
            alpha = e / np.sqrt(cVnic)
            #beta  = alpha * (alpha + cmni / cVnic);
            #r     = beta * cVnic / (1 - cVnic * beta);
            beta  = alpha * (alpha * cVnic + cmni)
            r     = beta / (1 - beta)
            # new message
            pnew  = r / cVnic
            mpnew = r * ( alpha + cmni / cVnic ) + alpha
        
            # update terms
            dp    = np.max([-p + eps,gamma * (pnew - p)]) # at worst, remove message
            dmp   = np.max([-mp + eps,gamma * (mpnew- mp)])
            d     = np.max([dmp,dp]) # for convergence measures
        
            pnew  = p  + dp;
            mpnew = mp + dmp;
            #project out to marginal
            Vnew  = V -  dp / (1 + dp * cVc) *np.outer(Vc,Vc)
            
            Mnew  = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc
            if np.any(np.isnan(Vnew)): raise Exception("oo")
            #if np.i Vnew)); keyboard; end
            #% if z < -30; keyboard; end
        
            #% normalization constant
            #%logS  = lP - 0.5 * (log(beta) - log(pnew)) + (alpha * alpha) / (2*beta);
            #% there is a problem here, when z is very large
            logS  = lP - 0.5 * (np.log(beta) - np.log(pnew) - np.log(cVnic)) + (alpha * alpha) / (2*beta) * cVnic
             
        elif exit_flag == -1:
            d = np.NAN
            Mnew  = 0
            Vnew  = 0;
            pnew  = 0;    
            mpnew = 0;
            logS  = -np.Infinity;
            #raise Exception("-----")
        elif exit_flag == 1:
            d     = 0
            # remove message from marginal:
            # new message
            pnew  = 0 
            mpnew = 0
        
            # update terms
            dp    = -p # at worst, remove message
            dmp   = -mp
            d     = max([dmp, dp]); # for convergence measures
        
            # project out to marginal
            Vnew  = V - dp / (1 + dp * cVc) * (np.outer(Vc,Vc))
            Mnew  = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc;
        
            logS  = 0;
        return Mnew,Vnew,pnew,mpnew,logS,d
    
    def _log_relative_gauss(self, z):
        if z < -6:
            return 1, -1.0e12, -1
        if z > 6:
            return 0, 0, 1 
        else:
            logphi = -0.5 * ( z * z + l2p)
            logPhi = np.log(.5 * scipy.special.erfc(-z / sq2))
            e = np.exp(logphi - logPhi)
            return e, logPhi, 0
