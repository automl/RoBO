#encoding=utf8
"""
this module contains acquisition functions that have high values
where the objective function is low.

.. class:: AcquisitionFunction

    An acquisition function is a class that gets instatiated with a model 
    and optional additional parameters. It then gets called via a maximizer.

    .. method:: __init__(model, **optional_kwargs)
                
        :param model: A model should have at least the function getCurrentBest() 
                      and predict(X, Z)

    .. method:: __call__(X, Z=None)
               
        :param X: X values, where to evaluate the acquisition function 
        :param Z: instance features to evaluate at. Could be None.
    
    .. method:: model_changed()
    
        this method should be called if the model is updated. The Entropy search needs
        to update its aproximation about P(x=x_min) 
"""
from scipy.stats import norm
import scipy
import numpy as np

class PI(object):
    def __init__(self, model, par=0.001, **kwargs):
        self.model = model
        self.par = par
    def __call__(self, X, Z=None, **kwargs):
        mean, var = self.model.predict(X, Z)
        Y_star = self.model.getCurrentBest()
        u = norm.cdf((Y_star - mean - self.par ) / var)
        return u
    def model_changed(self):
        pass

class UCB(object):
    def __init__(self, model, par=1.0, **kwargs):
        self.model = model
        self.par = par
    def __call__(self, X, Z=None, **kwargs):
        mean, var = self.model.predict(X, Z)
        return -mean + self.par * var
    def model_changed(self):
        pass


sq2 = np.sqrt(2)
l2p = np.log(2) + np.log(np.pi)
eps = np.finfo(np.float32).eps
debug_print = False

class Entropy(object):
    def __init__(self, model, X_lower, X_upper, **kwargs):
        self.model = model
        self.Nb = 10 
        self.X_lower = np.array(X_lower)
        self.X_upper = np.array(X_upper)
        self.UCB = UCB(model)
        
    def __call__(self, X, Z=None, **kwargs):
        return self.UCB(X, Z, **kwargs)
    
    def model_changed(self):
        #self.zb = np.add(np.multiply((self.X_upper - self.X_lower), np.random.uniform(size=(self.Nb, self.X_lower.shape[0]))), self.X_lower)
        self.zb = np.zeros((self.Nb, self.X_lower.shape[0]))
        for i in range(self.X_lower.shape[0]):
            self.zb[:,i] = np.linspace(self.X_lower[i], self.X_upper[i], self.Nb, endpoint = False)
        self.mb = np.dot(-np.log(np.prod(self.X_upper - self.X_lower)), np.ones((self.Nb, 1)))
        mu, var = self.model.predict(np.array(self.zb), full_cov=True)
        
        self.logP,self.dlogPdMu,self.dlogPdSigma,self.dlogPdMudMu = self._joint_min(mu, var, with_derivatives=True)           
        
    def _joint_min(self, mu, var, with_derivatives= False, **kwargs):
        
        #fac = 42.9076/68.20017903
        old_var = np.copy(var)
        fac = 1.0
        var = fac * var
        
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
        """
        1: Initialise with any q(x) defined by Z, μ, Σ (typically the parameters of p 0 (x)).
        2: Initialise messages t  ̃ i with zero precision.
        3: while q(x) has not converged do
        4:     for i ← 1 : m do
        5:        form cavity local q \i (x) by Equation 17 (stably calculated by 51).
        6:        calculate moments of q \i (x)t i (x) by Equations 21-23.
        7:        choose t  ̃ i (x) so q \i (x) t  ̃ i (x) matches above moments by Equation 16.
        8:     end for
        9:     update μ, Σ with new t  ̃ i (x) (stably using Equations 53 and 58).
        10:end while
        11:calculate Z, the total mass of q(x) using Equation 19 (stably using Equation 60).
        12:return Z, the approximation of F (A).
        """
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
            mpm = None
            
            mpm     = MP * MP / P;
            if not all(MP != 0):
                mpm[np.where(MP ==0)]=0
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
        """
        1: Initialise with any q(x) defined by Z, μ, Σ (typically the parameters of p 0 (x)).
        2: Initialise messages t  ̃ i with zero precision.
        3: while q(x) has not converged do
        4:     for i ← 1 : m do
        5:        form cavity local q \i (x) by Equation 17 (stably calculated by 51).
        6:        calculate moments of q \i (x)t i (x) by Equations 21-23.
        7:        choose t  ̃ i (x) so q \i (x) t  ̃ i (x) matches above moments by Equation 16.
        8:     end for
        9:     update μ, Σ with new t  ̃ i (x) (stably using Equations 53 and 58).
        10:end while
        11:calculate Z, the total mass of q(x) using Equation 19 (stably using Equation 60).
        12:return Z, the approximation of F (A).
        """
        """
        c_i = delta_{il} - delta{is}
        """
        """
        Equation 17:
        μ_\i = σ_\i^2 (c^T_i*μ          μ̃ _i )
                      (-----------  -   -----
                      (c_i^T Σ c_i      σ̃ i^2)
                      
        σ_\i^2 = ((c^T_i Σ c_i)^-1 - σ̃ i^-2)^-1
        
        Equation 11:
        q \i = q / t̃ _i = Z \i N(x;u \i, V\i)
        
        Equation 21:
        
        """
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


    # This method corresponds to the function SampleBeliefLocations in the original ES code
    # It is assumed that the GP data structure is a Python dictionary
    # This function calls PI, EI etc and samples them (using their values)
    def sample_from_measure(self, gp, xmin, xmax, n_representers, BestGuesses, acquisition_fn):
        # If there are no prior observations, do uniform sampling
        if (gp['x'].size == 0):
            dim = xmax.size
            zb = np.add(np.multiply((xmax - xmin), np.random.uniform(size=(n_representers, dim))), xmin)
            mb = np.dot(-np.log(np.prod(xmax - xmin)), np.ones((n_representers, 1)))
            return zb, mb

        # There are prior observations, i.e. it's not the first ES iteration
        dim = gp['x'].shape[1]

        # Calculate the step size for the slice sampler
        d0 = np.divide(
            np.linalg.norm((xmax - xmin), ord = 2),
            2)

        # zb will contain the sampled values:
        zb = np.zeros((n_representers, dim))
        mb = np.zeros((n_representers, 1))

        # Determine the number of batches for restarts
        numblock = np.floor(n_representers / 10.)
        restarts = np.zeros((numblock, dim))

        ### I don't really understand what the idea behind the following two assignments is...
        restarts[0:(np.minimum(numblock, BestGuesses.shape[0])), ] = \
            BestGuesses[np.maximum(BestGuesses.shape[0]-numblock+1, 1) - 1:, ]

        restarts[(np.minimum(numblock, BestGuesses.shape[0])):numblock, ] = \
            np.add(xmin,
                   np.multiply((xmax-xmin),
                               np.random.uniform(
                                   size = (np.arange(np.minimum(numblock, BestGuesses.shape[0]) + 1, numblock + 1).size, dim)
                               )))

        xx = restarts[0, ]
        subsample = 20 # why this value?
        for i in range(0, subsample * n_representers + 1): # Subasmpling by a factor of 10 improves mixing (?)
            
            if (i % (subsample*10) == 0) & (i / (subsample*10.) < numblock):
                xx = restarts[i/(subsample*10), ]
                # print str(xx)
            xx = self.slice_ShrinkRank_nolog(xx, acquisition_fn, d0)
            if i % subsample == 0:
                zb[(i / subsample) - 1, ] = xx
                emb = acquisition_fn(xx)
                mb[(i / subsample) - 1]  = np.log(emb)

        # Return values
        return zb, mb

    def projNullSpace(self, J, v):
        # Auxiliary function for the multivariate slice sampler
        if J.shape[1] > 0:
            return v - J.dot(J.transpose()).dot(v)
        else:
            return v

    def slice_shrinkRank_nolog(self, xx, P, s0, transpose):
        # This function is equivalent to the similarly named function in the original ES code
        if transpose:
            xx = xx.transpose()

        D = xx.shape[0]
        f = P(xx.transpose())[0]
        logf = np.log(f)
        logy = np.log(np.random.uniform()) + logf

        theta = 0.95

        k = 0
        s = s0
        c = np.zeros((D,0))
        J = np.zeros((0,0))

        while True:
            k += 1
            c = np.append(c, self.projNullSpace(J, xx + s[k-1] * np.random.normal(size=(D,1))))
            sx = np.divide(1., np.sum(np.divide(1., s)))
            mx = np.dot(
                sx,
                np.sum(
                    np.multiply(
                        np.divide(1., s),
                        np.subtract(c, xx)
                    ),
                    1))
            xk = xx + self.projNullSpace(J, mx + np.multiply(sx, np.random.normal(size=(D,1))))

            fk, dfk = P(xk.transpose())
            logfk  = np.log(fk)
            dlogfk = np.divide(dfk, fk)

            if logfk > logy: # accept these values
                xx = xk.transpose()
                return xx
            else: # shrink
                g = self.projNullSpace(J, dlogfk)
                if J.shape[1] < D - 1 & \
                   np.dot(g.transpose(), dlogfk) > 0.5 * np.linalg.norm(g) * np.linalg.norm(dlogfk):
                    J = np.append(J, np.divide(g, np.linalg.norm(g)))
                    s[k] = s[k-1]
                else:
                    s[k] = np.multiply(theta, s[k-1])
                    if s[k] < np.spacing(1):
                        print 'bug found: contracted down to zero step size, still not accepted.\n'
                    if transpose:
                        xx = xx.transpose()
                        return xx
                    else:
                        return xx


class EI(object):
    def __init__(self, model, par = 0.01, **kwargs):
        self.model = model
        self.par = par
    def __call__(self, x, Z=None, **kwargs):
        f_est = self.model.predict(x)
        eta = self.model.getCurrentBest()
        z = (eta - f_est[0] +self.par) / f_est[1]
        acqu = (eta - f_est[0] +self.par) * norm.cdf(z) + f_est[1] * norm.pdf(z)
        return acqu

    def model_changed(self):
        pass


class LogEI(object):
    def __init__(self, model, par = 0.01):
        self.model = model
        self.par = par
    def __call__(self, x, par = 0.01, Z=None, **kwargs):
        f_est = self.model.predict(x)
        eta = self.model.getCurrentBest()
        z = (eta - f_est[0] - self.par) / f_est[1]
        log_ei = np.zeros((f_est[0].size, 1))

        for i in range(0, f_est[0].size):
            mu, sigma = f_est[0][i], f_est[1][i]
            # Degenerate case 1: first term vanishes
            if abs(eta - mu) == 0:
                if sigma > 0:
                    log_ei[i] = np.log(sigma) + norm.logpdf(z[i])
                else:
                    log_ei[i] = -float('Inf')
            # Degenerate case 2: second term vanishes and first term has a special form.
            elif sigma == 0:
                if mu < eta:
                    log_ei[i] = np.log(eta - mu)
                else:
                    log_ei[i] = -float('Inf')
            # Normal case
            else:
                b = np.log(sigma) + norm.logpdf(z[i])
                # log(y+z) is tricky, we distinguish two cases
                if eta > mu:
                    # When y>0, z>0, we define a=ln(y), b=ln(z).
                    # Then y+z = exp[ max(a,b) + ln(1 + exp(-|b-a|)) ],
                    # and thus log(y+z) = max(a,b) + ln(1 + exp(-|b-a|))
                    a = np.log(eta - mu) + norm.logcdf(z[i])
                    log_ei[i] = max(a, b) + np.log(1 + np.exp(-abs(b-a)))
                else:
                    # When y<0, z>0, we define a=ln(-y), b=ln(z), and it has to be true that b >= a in order to satisfy y+z>=0.
                    # Then y+z = exp[ b + ln(exp(b-a) -1) ],
                    # and thus log(y+z) = a + ln(exp(b-a) -1)
                    a = np.log(mu - eta) + norm.logcdf(z[i])
                    if a >= b:
                        # a>b can only happen due to numerical inaccuracies or approximation errors
                        log_ei[i] = -float('Inf')
                    else:
                        log_ei[i] = b + np.log(1 - np.exp(a - b))
        return log_ei

    def model_changed(self):
        pass    
if __name__ == "__main__":
    Var= np.array([[  3.87161361e+02,   3.13109917e+02,   1.69058548e+02,   4.72396572e+01,   -4.04833571e-01,   6.72054465e+00,   2.18906542e+01,   2.20972808e+01,    1.29360166e+01,   5.03719525e+00],
       [3.13109917e+02, 3.23248412e+02, 2.14223648e+02, 7.13191223e+01, -3.97504818e-01, 1.17772024e+01, 4.18066141e+01, 4.41174447e+01, 2.65510341e+01, 1.05254731e+01],
       [1.69058548e+02, 2.14223648e+02, 1.72063217e+02, 6.71816966e+01, 1.86713858e-01, 1.20431800e+01, 4.83602238e+01, 5.42833203e+01, 3.40629676e+01, 1.39084617e+01],
       [4.72396572e+01, 7.13191223e+01, 6.71816966e+01, 3.25011845e+01, 9.87398227e-01, 4.72609238e+00, 2.46312918e+01, 3.05316334e+01, 2.05150727e+01, 8.82510182e+00],
       [-4.04833571e-01, -3.97504818e-01, 1.86713858e-01, 9.87398227e-01, 1.80923222e+00, -7.13587511e-01, -2.20417380e+00, -2.36386527e+00, -1.51086496e+00, -6.43132255e-01],
       [6.72054465e+00, 1.17772024e+01, 1.20431800e+01, 4.72609238e+00, -7.13587511e-01, 8.36646407e+00, 2.28136296e+01, 2.88246428e+01, 2.15302329e+01, 1.05999706e+01],
       [2.18906542e+01, 4.18066141e+01, 4.83602238e+01, 2.46312918e+01, -2.20417380e+00, 2.28136296e+01, 9.75746326e+01, 1.46711508e+02, 1.29666502e+02, 7.57110465e+01],
       [2.20972808e+01, 4.41174447e+01, 5.42833203e+01, 3.05316334e+01, -2.36386527e+00, 2.88246428e+01, 1.46711508e+02, 2.64285998e+02, 2.79585614e+02, 2.00199662e+02],
       [1.29360166e+01, 2.65510341e+01, 3.40629676e+01, 2.05150727e+01, -1.51086496e+00, 2.15302329e+01, 1.29666502e+02, 2.79585614e+02, 3.68458082e+02, 3.34272268e+02],
       [5.03719525e+00, 1.05254731e+01, 1.39084617e+01, 8.82510182e+00, -6.43132255e-01, 1.05999706e+01, 7.57110465e+01, 2.00199662e+02, 3.34272268e+02, 3.96936615e+02]])
    mu= np.array([  -5.49136396, -7.12505187, 4.35396108, 40.52508471, 91.58672473, 123.45181813, 112.56542119, 72.53546694, 33.71330893, 11.42921143])
    e =  Entropy(None, None, None)
    
    
    a, b, c, d = e._joint_min(mu, Var, with_derivatives = True)
