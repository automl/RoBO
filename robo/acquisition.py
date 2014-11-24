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
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        raise NotImplementedError
    def _ep_pmin(self, X, Z=None, with_derivatives= False, **kwargs):
        
        mu, var = self.model.predict(np.array(zb), full_cov=True)
        fac = 42.9076/68.20017903
        var = fac * var
        logP = np.empty(mu.shape)
        #for i ← 1 : m do
        D = mu.shape[0]
        dlogPdMu    = np.zeros((D,D));
        dlogPdSigma = np.zeros((D,0.5 * D * (D+1)));
        dlogPdMudMu= np.zeros((D,D,D));
        for i in xrange(mu.shape[0]):
            #logP[k] ) self._min_faktor(mu, var, 0)
            a = self._min_faktor(mu, var, i)
            logP[i] = a.next()
            dlogPdMu[i,:] = np.transpose(a.next())
            dlogPdMudMu[i, :, :] = a.next()
            dlogPdSigma[i,:] = np.transpose(a.next())
            
        logP[np.isinf(logP)] = -500;    
        #re-normalize at the end, to smooth out numerical imbalances:
        logPold        = logP
        dlogPdMuold    = dlogPdMu
        dlogPdSigmaold = dlogPdSigma
        dlogPdMudMuold = dlogPdMudMu;
        Z     = np.sum(np.exp(logPold));
        _maxLogP = np.max(logP)
        s = _maxLogP + np.log(np.sum(np.exp(logP - _maxLogP)))
        s = _maxLogP if np.isinf(s) else s
        
        logP  = logP - s;
        
        # adjust derivatives, too. This is a bit tedious.
        Zm    = sum(np.rot90((np.exp(logPold)*np.rot90(dlogPdMuold,1)),3)) / Z
        Zs    = sum(np.rot90((np.exp(logPold)*np.rot90(dlogPdSigmaold,1)),3)) / Z 
        
        
        dlogPdMu    = dlogPdMuold-Zm
        dlogPdSigma = dlogPdSigmaold - Zs
    
        #ff   = dlogPdMuold' * diag(exp(logPold)) * dlogPdMuold ./ Z;    
        #gg   = etprod('ij',dlogPdMudMuold,'kij',exp(logPold),'k') ./ Z;
        #ff = np.tensordot(dlogPdMuold,dlogPdMuold)
        ff = np.einsum('ki,kj->kij', dlogPdMuold, dlogPdMuold)
        
        
        gg   = np.einsum('kij,k->ij',dlogPdMudMuold+ff,np.exp(logPold)) / Z;
        print "gg= ",gg
        Zij  = Zm.T * Zm;
        adds = np.reshape(-gg+Zij,(1,D,D));
        print adds[:,:49]
        #dlogPdMudMu = bsxfun(@plus,dlogPdMudMuold,adds);"""
        #print "logP = ", logP
        
        #print "dlogPdMudMu = ", dlogPdMudMu[12,12,:]
        #print "dlogPdSigma = ", dlogPdSigma
            
        
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
                M, V, P[i], MP[i], logS[i], d = self._lt_factor(k, l, M, V, MP[i], P[i], gamma)
                
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
            dlogZdSigma = np.zeros((0.5*(D*(D+1)),1))
            yield dlogZdSigma
            dlogZdMudMu = np.zeros((D,D))
            yield dlogZdMudMu
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
            
            R       = np.sqrt(np.transpose(P)) * C
            r       = np.sum(np.transpose(MP) * C, 1)
            mpm     = MP * MP / P;
            mpm     = sum(mpm);
            
            s       = sum(logS);
            IRSR    = (np.eye(D-1) + np.dot(np.dot(np.transpose(R) , Sigma), R));
            rSr     = np.dot(np.dot(np.transpose(r), Sigma) , r);
            A =  np.dot(R,np.linalg.solve(IRSR,np.transpose(R))) 
            
            A       = 0.5 * (np.transpose(A) + A) # ensure symmetry.
            b       = (Mu + np.dot(Sigma,r));
            Ab      = np.dot(A,b);
            dts     = 2 * np.sum(np.log(np.diagonal(np.linalg.cholesky(IRSR))));
            logZ    = 0.5 * (rSr - np.dot(np.transpose(b), Ab) - dts) + np.dot(np.transpose(Mu), r) + s - 0.5 * mpm;
            yield logZ
            btA = np.dot(np.transpose(b), A)
            
            dlogZdMu    = r - Ab
            yield dlogZdMu
            dlogZdMudMu = -A
            yield dlogZdMudMu
            dlogZdSigma = -A - 2*np.outer(r,np.transpose(Ab)) + np.outer(r,np.transpose(r)) + np.outer(np.transpose(btA),np.transpose(Ab));
            _dlogZdSigma = np.zeros_like(dlogZdSigma)
            np.fill_diagonal(_dlogZdSigma, np.diagonal(dlogZdSigma))
            dlogZdSigma = 0.5*(dlogZdSigma+np.transpose(dlogZdSigma)-_dlogZdSigma)
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
            
            #print "[---\n",Vc * np.transpose(Vc),"\n---\n", np.dot(Vc,np.transpose(Vc)),"\n---]"
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
            raise Exception("-----")
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
    def model_changed(self):
        raise NotImplementedError

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
            # print i,
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
        

#possible x values sampled via EI

#log EI values

#Expectation of zb of GP

#logP

def test():
    import GPy
    from models import GPyModel
    from test_functions import branin
    kernel = GPy.kern.rbf(input_dim=2, variance=6.5816*6.5816, lengthscale=[5.9076, 5.9076], ARD=True)
    X_lower = np.array([-8,-8])
    X_upper = np.array([19, 19])
    X = np.empty((1, 2))
    Y = np.empty((1, 1))
    X[0,:] = [2.6190,    5.4830] #random.random() * (X_upper[0] - X_lower[0]) + X_lower[0]];
    
    objective_fkt= branin
    
    Y[0:] = objective_fkt(X)
    model = GPyModel(kernel,noise_variance=0.044855*0.044855)
    model.train(X,Y)
    model.m.optimize()
    #print model.predict(np.array([[0.3, 0.4],[0.2, 0.5]]), full_cov=True)
    #print kernel.K(np.array([[0.3, 0.4],[9.2, 0.5]]))* 42.9076/68.20017903
    e = Entropy(model)
    e._ep_pmin(zb)
    #print model.m
    #print model.predict(np.array(zb[0:2,:]))
    

if __name__ == "__main__":
    zb =np.array([[14.3455,    0.4739],
        [4.7943,   -2.2227],
        [5.0096,   -3.8949],
       [-2.5391,    3.5426],
       [-0.5479,    1.7762],
       [-2.6867,    4.1843],
       [-2.4620,   12.2733],
       [-0.0505,   -3.2885],
        [2.7611,   -4.8736],
       [13.9763,    8.7406],
        [8.5846,   13.7285],
        [8.8334,   -0.8107],
        [4.4899,    3.0050],
       [14.6994,   10.2945],
       [12.2605,    2.0057],
       [13.0723,   10.5736],
        [9.1154,   -0.3920],
        [3.1969,   13.8373],
        [5.8514,    5.3512],
       [10.6769,   10.6335],
       [13.2543,    0.3430],
        [8.3957,    8.9913],
        [4.8477,   14.5783],
        [7.6631,   14.2574],
       [-1.1363,   -4.0688],
       [-2.3865,   -1.8635],
       [-4.6856,   -2.9748],
       [13.2557,   -3.2124],
        [2.3016,   -4.2264],
        [9.4008,   -3.2105],
       [-3.8363,   11.6387],
       [13.4973,    1.3641],
       [-4.7570,   -3.1846],
       [14.0831,    5.2030],
       [14.7648,    7.1242],
        [1.3154,   -2.1135],
       [12.2356,    8.6618],
       [-3.1422,   -0.8687],
       [10.6497,   10.0474],
       [-4.5171,   11.1157],
        [3.3472,   -2.1601],
        [2.6074,   -0.7084],
        [7.9226,   -2.9915],
       [-3.0507,    2.4470],
       [-1.1147,   12.1506],
        [6.3822,   -4.8960],
        [7.2049,   -1.7246],
        [0.1262,   14.3814],
        [13.8807,   4.5143],
        [7.9338,    0.5726],
    ])
    
    lmb =np.array([
        [2.1841],
        [1.8634],
        [2.0266],
        [1.4540],
        [1.3051],
        [1.4434],
        [1.9170],
        [1.9842],
        [2.0748],
        [2.1509],
        [2.0629],
        [1.9540],
        [0.7383],
        [2.1914],
        [2.0677],
        [2.1428],
        [1.9456],
        [1.9055],
        [0.7901],
        [2.0176],
        [2.1507],
        [1.6886],
        [2.0012],
        [2.0591],
        [2.0687],
        [1.9583],
        [2.1216],
        [2.2093],
        [2.0295],
        [2.1138],
        [1.9612],
        [2.1431],
        [2.1316],
        [2.1357],
        [2.1678],
        [1.8261],
        [2.0596],
        [1.9269],
        [1.9902],
        [1.9772],
        [1.8222],
        [1.5904],
        [2.0504],
        [1.6336],
        [1.8176],
        [2.1146],
        [1.9235],
        [1.9905],
        [2.1279],
        [1.7617],
    ])
    """
    Mb =
    
        [0.7382]
        [3.4305]
        [2.1637]
        [5.8063]
        [6.4360]
        [5.8548]
        [3.0368]
        [2.5130]
        [1.7496]
        [1.0568]
        [1.8536]
        [2.7524]
        [8.0227]
        [0.6672]
        [1.8115]
        [1.1321]
        [2.8180]
        [3.1232]
        [7.9201]
        [2.2393]
        [1.0580]
        [4.5731]
        [2.3748]
        [1.8870]
        [1.8034]
        [2.7185]
        [1.3294]
        [0.4910]
        [2.1398]
        [1.4001]
        [2.6957]
        [1.1301]
        [1.2365]
        [1.1991]
        [0.8956]
        [3.6917]
        [1.8825]
        [2.9615]
        [2.4640]
        [2.5690]
        [3.7182]
        [5.1282]
        [1.9616]
        [4.8915]
        [3.7501]
        [1.3932]
        [2.9873]
        [2.4613]
        [1.2711]
        [4.1206]
    
    
    logP =
    
       [-2.9708]
       [-5.7440]
       [-4.8360]
       [-9.0724]
       [-8.3878]
       [-5.6195]
       [-4.3381]
       [-5.1012]
       [-3.1954]
       [-4.6078]
       [-3.3173]
       [-5.5302]
      [-10.7671]
       [-2.6518]
       [-4.3479]
       [-3.7004]
       [-4.9394]
       [-4.5230]
       [-9.7196]
       [-4.3727]
       [-4.4313]
       [-5.5980]
       [-3.2950]
       [-3.2949]
       [-3.3316]
       [-5.1482]
       [-3.7303]
       [-2.4316]
       [-4.9547]
       [-3.5023]
       [-4.0596]
       [-4.6006]
       [-2.5968]
       [-3.9474]
       [-3.1956]
       [-5.8456]
       [-4.5595]
       [-4.4042]
       [-5.2002]
       [-2.8613]
       [-6.0557]
       [-7.2198]
       [-4.8528]
       [-4.5403]
       [-5.6116]
       [-2.9593]
       [-5.5145]
       [-3.0241]
       [-3.8651]
       [-5.7802]
    
    """
    
        
    test()

