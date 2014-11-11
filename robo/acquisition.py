#encoding=utf8

from scipy.stats import norm
import scipy
import numpy as np
        
class PI(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        mean, var = self.model.predict(X, Z)
        Y_star = self.model.getCurrentBest()
        u = 1 - norm.cdf((mean - Y_star) / var)
        return u

class UCB(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        mean, var = self.model.predict(X, Z)
        return -mean + var
sq2 = np.sqrt(2)
l2p = np.log(2) + np.log(np.pi)
eps = np.finfo(np.float32).eps
class Entropy(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        raise NotImplementedError
    def _ep_pmin(self, X, Z=None, with_derivatives= False, **kwargs):
        
        mu, var = self.model.predict(np.array(zb))
        fac = 42.9076/68.20017903
        var = fac * var
        logP = np.empty(mu.shape)
        #for i ← 1 : m do
        for i in xrange(mu.shape[0]):
            self._min_faktor(mu, var, i)
        
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
        logS = np.empty((D-1,))
        #mean time first moment
        MP = np.empty((D-1,))
        #precision, second moment 
        P = np.empty((D-1,))
        
        M = np.copy(Mu)
        V = np.copy(Sigma)
        b = False
        for count in xrange(50):
            diff = 0
            for i in range(D-1):
                l = i if  i < k else i+1
                M, V, P[i], MP[i], logS[i], d = self._lt_factor(k, l, Mu, Sigma, MP[i], P[i], gamma)
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
            dlogZdMu = np.zeros((D,1))
            dlogZdSigma = np.zeros((0.5*(D*(D+1)),1))
            dlogZdMudMu = np.zeros((D,D))
            mvmin = [Mu[k],Sigma[k,k]]
            dMdMu = np.zeros((1,D))
            dMdSigma = np.zeros((1,0.5*(D*(D+1))))
            dVdSigma = np.zeros((1,0.5*(D*(D+1))))
        else:
            #evaluate log Z:
            """"C = eye(D) / sq2 
            C[k,:] = -1/sq2
            C[:,k] = []
            R       = sqrt(P') * C
            r       = sum(bsxfun(@times,MP',C),2);
            mpm     = MP.* MP ./ P;
            mpm(MP==0) = 0;
            mpm     = sum(mpm);
            s       = sum(logS);
            
            IRSR    = (eye(D-1) + R' * Sigma * R);
            rSr     = r' * Sigma * r;
            A       = R * (IRSR \ R');
            A       = 0.5 * (A' + A); % ensure symmetry.
            b       = (Mu + Sigma * r);
            Ab      = A * b;
            dts     = logdet(IRSR);
            logZ    = 0.5 * (rSr - b' * Ab - dts) + Mu' * r + s - 0.5 * mpm;
            if(logZ == inf); keyboard; end"""
            pass
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
        print "*-"*30
        print M
        print "*-"*30
        print V
        print "*-"*30
        cVc = (V[l,l] - 2*V[s,l] + V[s,s])/ 2
        Vc  = (V[:, l] - V [:, s]) / sq2
        cM =  (M[l] - M[s])/ sq2
        cVnic = np.max([cVc/(1-p * cVc), 0])
        cmni = cM + cVnic * (p * cM - mp)
        print "cVc = ", cVc
        print "Vc = ", Vc
        print "cM = ", cM
        print "cVnic =",cVnic
        print "cmni = ",cmni
        
        z     = cmni / np.sqrt(cVnic);
        
        e,lP,exit_flag = self._log_relative_gauss( z)
        print "e = ", e
        if exit_flag == 0:
            alpha = e / np.sqrt(cVnic)
            print "alpha =", alpha 
            #beta  = alpha * (alpha + cmni / cVnic);
            #r     = beta * cVnic / (1 - cVnic * beta);
            beta  = alpha * (alpha * cVnic + cmni)
            r     = beta / (1 - beta)
            print "r = ", r
            # new message
            pnew  = r / cVnic
            mpnew = r * ( alpha + cmni / cVnic ) + alpha
        
            # update terms
            dp    = np.max([-p + eps,gamma * (pnew - p)]) # at worst, remove message
            dmp   = np.max([-mp + eps,gamma * (mpnew- mp)])
            d     = np.max([dmp,dp]) # for convergence measures
        
            pnew  = p  + dp;
            mpnew = mp + dmp;
            print "pnew = ", pnew
            print "mpnew = ", mpnew
            #project out to marginal
            print "tmp1= ", dp / (1 + dp * cVc) 
            print "tmp2= ", np.mat(Vc) * np.mat(np.transpose(Vc))
            Vnew  = V -  dp / (1 + dp * cVc) * (Vc* np.transpose(Vc))
            #print "[---\n",Vc * np.transpose(Vc),"\n---\n", np.dot(Vc,np.transpose(Vc)),"\n---]"
            print "Vnew = ", Vnew 
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
            Vnew  = V - dp / (1 + dp * cVc) * (Vc * np.transpose(Vc));
            Mnew  = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc;
        
            logS  = 0;
        raise Exception
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
    
class EI(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        self.model.predict(zb)
        

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
    Y[0,:] = objective_fkt(X[0,:])
    
    model = GPyModel(kernel,noise_variance=0.044855)
    model.train(X,Y)
    model.m.optimize()
    e = Entropy(model)
    e._ep_pmin(zb)
    #print model.m
    #print model.predict(np.array(zb[0:2,:]))
    


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