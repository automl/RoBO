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
    
    .. method:: update(model)
    
        this method should be called if the model is updated. The Entropy search needs
        to update its aproximation about P(x=x_min) 
"""
import sys
from scipy.stats import norm
import scipy
import numpy as np
from robo.loss_functions import logLoss
from robo import BayesianOptimizationError 
from .LogEI import LogEI
from .PI import PI
class EI(object):
    def __init__(self, model, X_lower, X_upper, par = 0.01,**kwargs):
        self.model = model
        self.par = par
        self.X_lower = X_lower
        self.X_upper = X_upper
        self._alpha = None

    @property
    def alpha(self):
        invalid_alpha = (self._alpha is None or len(self.model.X) > len(self._alpha))
        valid_X = self.model.X is not None and len(self.model.X) > 0 
        if valid_X and invalid_alpha:
            self._alpha = np.linalg.solve(self.model.cK, np.linalg.solve(self.model.cK.transpose(), self.model.Y))
        elif self._alpha is None:
            raise Exception("self.model.X is not properly initialized in acquisition EI")
        return self._alpha

    def __call__(self, x, Z=None, derivative=False, verbose=False, **kwargs):
        if (x < self.X_lower).any() or (x > self.X_upper).any():
            if derivative:
                f = 0
                df = np.zeros((x.shape[1],1))
                return f, df
            else:
                return 0

        dim = x.shape[1]
        f_est = self.model.predict(x)
        
        eta = self.model.predict(np.array([self.model.getCurrentBestX()]))[0]
        z = (eta - f_est[0] + self.par) / f_est[1]
        
        f = (eta - f_est[0] + self.par) * norm.cdf(z) + f_est[1] * norm.pdf(z)
        if verbose:
            print "f = ", f
            print "s = ", f_est[1]
            print "eta = ", eta
            print "z = ",z
        if derivative:
            # Derivative of kernel values:
            dkxX = self.model.kernel.gradients_X(np.array([np.ones(len(self.model.X))]), self.model.X, x)
            dkxx = self.model.kernel.gradients_X(np.array([np.ones(len(self.model.X))]), self.model.X)

            # dm = derivative of the gaussian process mean function
            dmdx = np.dot(dkxX.transpose(), self.alpha)
            # ds = derivative of the gaussian process covariance function
            dsdx = np.zeros((dim, 1))
            
            for i in range(0, dim):
                dsdx[i] = np.dot(0.5 / f_est[1], dkxx[0,dim-1] - 2 * np.dot(dkxX[:,dim-1].transpose(),
                                                                            np.linalg.solve(self.model.cK,
                                                                                            np.linalg.solve(self.model.cK.transpose(),
                                                                                                            self.model.K[0,None].transpose()))))
        
            df = -dmdx * norm.cdf(z) + dsdx * norm.pdf(z)
        if (f < 0).any() :
            print f_est
            #print f
            #print df[np.where(f < 0)]
            #print "\n x (f<0)= ",
            #print x
            #print "\n z (f<0)= \n", 
            #print z[np.where(f < 0)] 
            #print "\n eta = \n" 
            #print eta
            #print "\n mean (f<0)= \n"  
            #print f_est[0][np.where(f < 0)] 
            #"\n mean (f>=0)= \n"  , 
            #f_est[0][np.where(f >= 0)], 
            #print "\n sigma (f<0)= \n",
            #print f_est[1][np.where(f < 0)] 
                   
            raise Exception("EI can't be smaller than 0")
        if derivative:
            return f, df
        else:
            return f
    def update(self, model):
        self.model = model

class UCB(object):
    def __init__(self, model, par=1.0, **kwargs):
        self.model = model
        self.par = par
    def __call__(self, X, Z=None, **kwargs):
        mean, var = self.model.predict(X, Z)
        return -mean + self.par * var
    def update(self, model):
        self.model = model

sq2 = np.sqrt(2)
l2p = np.log(2) + np.log(np.pi)
eps = np.finfo(np.float32).eps

class Entropy(object):
    def __init__(self, model, X_lower, X_upper, Nb = 100, sampling_acquisition = None, sampling_acquisition_kw = {"par":0.4}, T=200, loss_function=None, **kwargs):
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
        
    def __call__(self, X, Z=None, **kwargs):
        return self.dh_fun_true(X)[0]

    def update(self, model):
        self.model = model
        self.sampling_acquisition.update(model)
        self.zb, self.lmb = self.sample_from_measure(self.X_lower, self.X_upper, self.Nb, self.BestGuesses, self.sampling_acquisition)
        #if np.isinf(self.lmb).any():
        #    raise Exception("lmb is inf")
        mu, var = self.model.predict(np.array(self.zb), full_cov=True)
        self.logP,self.dlogPdMu,self.dlogPdSigma,self.dlogPdMudMu = self._joint_min(mu, var, with_derivatives=True)
        self.current_entropy = - np.sum (np.exp(self.logP) * (self.logP+self.lmb) )
        #self.acq = self.dh_mc_local(self.zb, self.logP,self.dlogPdMu,self.dlogPdSigma,self.dlogPdMudMu, self.T, self.lmb, self.X_lower, self.X_upper, False, self.loss_function )
        self.W = np.random.randn(1, self.T)
        #self.W = np.array([[-2.17462056629639, 0.144614651456100, 0.669033435302303, -0.848829159757920, -0.0570901191929413, 0.291786264353796, -0.614618540132659, -0.0136431885670602, 0.412113496148595, -0.937149595361607, 0.835157036407438, 1.09453846090965, 0.321180732968874, 0.0151128385440506, -1.16449254121656, 0.494840747422341, -1.21584898482425, 1.59078797675859, 0.526475048595550, 0.264818591763603, -0.393597890579411, -2.17631003833625, -1.19565443202559, 0.365328392431846, 2.41244305717062, 0.740247220579396, 0.309746388607370, -0.541987183113314, 0.122283184138499, 0.962279332770060, -0.253974843323265, 1.89175598926363, -1.22025413722763, -0.376237143988673, 0.0524291487772232, -1.95393370060853, 0.565420571160198, 0.0777944624002781, 0.0807011876263225, 0.794707809437465, 1.04368893647872, 0.802184648027535, -0.468824364835627, -0.0465923743265948, 0.353453034782426, -1.25680056318429, -1.03603143787954, -0.426962687763533, 0.0776686794320303, 1.79093025896060, 0.797263135406905, -0.442500740824704, -0.629091091514816, 1.53379627132611, 2.73354038131324, 0.167439077485047, -0.478865165602679, -1.41265388318180, 0.974988055068974, 0.289718984218535, 1.16540825109215, -0.908705616318882, 0.240575037229452, -0.654046335191623, 1.27495453830672, -0.551190610935942, -0.0874088682817872, -0.347820318259741, 0.993962534155598, 0.428436173023810, -1.00418528016197, -0.632687876663078, -1.04570999122494, 0.505707009458752, 0.331953721396756, -1.63541252268811, -1.90684020531619, -0.0403987210424983, 0.697193329646415, 0.168802575488670, -1.54379632573123, -1.55180239414170, 0.867077803619383, -0.145372990423397, -0.386241663817012, 1.31620411236280, -0.796464226593885, 0.135441371885595, 0.417806430208596, 0.819879645963700, -0.854353379120009, 0.357318588185117, 2.74846655587521, -1.51297699352819, 0.433979032773600, -0.229767733464022, -0.827086376299091, -0.831982615393436, 0.497899500583305, 2.31562610334450, -0.793825701652537, 0.540959569281976, -0.559061357111602, 1.97656605832667, 0.544660375289434, -0.137905557777941, 0.619875635750664, -0.00558278344490564, 1.10719924542420, -0.185590108437143, -1.12141781251190, 0.246449516399111, 1.56103723018519, -1.19662171456272, -0.242345277992204, 1.00482816344250, -1.92012580512738, 0.625445779460800, 0.752960417782594, 0.213483683522705, -0.770234072868573, -0.00713465252874278, 0.0931865247774352, 0.935251207390719, 0.663517921831603, -0.350232666735205, 1.61987743935497, -0.0508330964816129, -0.812697070766603, -0.438419934696927, 0.858609994635494, 0.195215935238338, 0.888862153882707, 0.0692218210392502, 2.48682132029734, -1.66564311788150, -0.415927944852021, -0.0842246064708947, 0.0892519475074891, 1.45614720047535, 0.219453518638082, -0.114877251919321, 0.0686024694830988, 0.751494856970101, -0.689427057923177, 0.450813584524417, -1.56502465785673, -0.0787959236624287, -0.941791969061035, -0.652861505816215, 0.282529239916988, -1.12507904298537, -0.988974087143185, -1.51585876104897, -2.22851047687315, -0.151521678303304, 1.16274436560636, -0.181936286155354, -0.233953540330954, -1.04673406988337, 1.58331887778704, -0.249413421524592, 1.29401900835506, 0.361980137920309, -0.263052508703654, 1.08214445191656, 0.982434080192891, -0.111443964288855, 1.29595078548944, 1.78129502150358, -0.0656148757390404, -0.255680865369883, 0.404638157039467, 0.351756753327098, 0.807511060208285, -0.255590492307781, 0.715113278785686, 1.04859731286360, 0.177548555016550, 0.153297684827984, -1.25432447831637, -1.17280085830884, -1.46144181507780, -1.24428798710640, -0.155099375334764, -1.61491381350160, 2.24606514978483, 0.739281132309719, -2.06764536263329, -1.15455058140430, -0.107191878127395, -1.71287908652886, -0.459894542126372, 1.09100010337293, -0.163880256756727, -0.0672404702443348, -1.00159245598463, -0.555661447134641, -1.35155634252200, 0.364211322734809]])
        self.K = self.model.K
        self.cK = self.model.cK.T
        self.kbX = self.model.kernel.K(self.zb,self.model.X)
        #self.L = self._gp_innovation_local(self.zb)
        self.logP = np.reshape(self.logP, (self.logP.shape[0], 1))

    def dh_fun_false(self,x, **kwargs):
        return self.dh_fun(x, False)

    def dh_fun_true(self,x, **kwargs):
        return self.dh_fun(x, True)

    def dh_fun(self,x, invertsign = True):
        logP = self.logP
        dlogPdM = self.dlogPdMu
        dlogPdV = self.dlogPdSigma
        ddlogPdMdM = self.dlogPdMudMu
        lmb = self.lmb 
        W = self.W 
        L = self._gp_innovation_local
        xmin = self.X_lower
        xmax = self.X_upper

        LossFunc = self.loss_function 
        zbel = self.zb
        # If x is a vector, convert it to a matrix (some functions are sensitive to this distinction)
        if len(x.shape) == 1:
            x = x[np.newaxis]
        if np.any(x < xmin) or np.any(x > xmax):
            dH = np.spacing(1)
            ddHdx = np.zeros((x.shape[1], 1))
            return dH, ddHdx
        if x.shape[0] > 1:
            raise BayesianOptimizationError(BayesianOptimizationError.SINGLE_INPUT_ONLY, "dHdx_local is only for single x inputs")
        
        # Number of belief locations:
        N = logP.size

        D = x.shape[0]
        T = W.shape[0]
        # Evaluate innovation
        Lx, _ = L(x)
        # Innovation function for mean:
        dMdx = Lx
        # Innovation function for covariance:
        dVdx = -Lx.dot(Lx.T)
        # The transpose operator is there to make the array indexing equivalent to matlab's
        dVdx = dVdx[np.triu(np.ones((N,N))).T.astype(bool), np.newaxis]

        dMM = dMdx.dot(dMdx.T)
        trterm = np.sum(np.sum(
            np.multiply(ddlogPdMdM, np.reshape(dMM, (1, dMM.shape[0],dMM.shape[1]))),
            2), 1)[:, np.newaxis]

        # add a second dimension to the arrays if necessary:
        logP = np.reshape(logP, (logP.shape[0], 1))
        # logP = np.reshape(logP, (logP.shape[0], 1))

        

        # Deterministic part of change:
        detchange = dlogPdV.dot(dVdx) + 0.5 * trterm
        # Stochastic part of change:
        stochange = (dlogPdM.dot(dMdx)).dot(W)
        # Predicted new logP:
        
        lPred = np.add(logP + detchange, stochange)
        #
        _maxLPred = np.max(lPred)
        s = _maxLPred + np.log(np.sum(np.exp(lPred - _maxLPred)))
        lselP = _maxLPred if np.isinf(s) else s
        
        #
        #lselP = np.log(np.sum(np.exp(lPred), 0))[np.newaxis,:]
        # Normalise:
        lPred = np.subtract(lPred, lselP)
        
        dHp = LossFunc(logP, lmb, lPred, zbel)
        dH = np.mean(dHp)

        if invertsign:
            dH = - dH
        if not np.isreal(dH):
            raise Exception("dH is not real")
        # Numerical derivative, renormalisation makes analytical derivatives unstable.
        e = 1.0e-5
        ddHdx = np.zeros((D,1))
        for d in range(D):
            ### First part:
            y = np.array(x)
            y[d] += e

            # Evaluate innovation:
            Ly, _ = L(y)
            # Innovation function for mean:
            dMdy = Ly
            # Innovation function for covariance:
            dVdy = -Ly.dot(Ly.T)
            dVdy = dVdy[np.triu(np.ones((N,N))).T.astype(bool), np.newaxis]

            dMM = dMdy.dot(dMdy.T)
            # TODO: is this recalculation really necessary? (See below as well)
            trterm = np.sum(np.sum(
                np.multiply(ddlogPdMdM, np.reshape(dMM, (1,dMM.shape[0],dMM.shape[1]))),
                2), 1)[:, np.newaxis]
            # trterm = np.array(
            #     [[-0.843010908566320], \
            #      [-0.582029313251303], \
            #      [-0.758961809411626], \
            #      [-0.534968553942584], \
            #      [-0.805424071511742], \
            #      [-0.206116671234491], \
            #      [-0.702439309633706], \
            #      [-0.704391419459464], \
            #      [-0.930137189649854], \
            #      [-0.916431159870648]])

            # Deterministic part of change:
            detchange = dlogPdV.dot(dVdy) + 0.5 * trterm
            # Stochastic part of change:
            stochange = (dlogPdM.dot(dMdy)).dot(W)
            # Predicted new logP:
            lPred = np.add(logP + detchange, stochange)
            _maxLPred = np.max(lPred)
            s = _maxLPred + np.log(np.sum(np.exp(lPred - _maxLPred)))
            lselP = _maxLPred if np.isinf(s) else s
            # Normalise:
            lPred = np.subtract(lPred, lselP)

            dHp = LossFunc(logP, lmb, lPred, zbel)
            dHy1 = np.mean(dHp, dtype=np.float64)

            ### Second part:
            y = np.array(x)
            y[d] = y[d] - e

            # Evaluate innovation:
            Ly, _ = L(y)
            # Innovation function for mean:
            dMdy = Ly
            # Innovation function for covariance:
            dVdy = -Ly.dot(Ly.T)
            dVdy = dVdy[np.triu(np.ones((N,N))).T.astype(bool), np.newaxis]

            dMM = dMdy.dot(dMdy.T)
            # TODO: is this recalculation really necessary? (See below as well)
            trterm = np.sum(np.sum(
                np.multiply(ddlogPdMdM, np.reshape(dMM, (1,dMM.shape[0],dMM.shape[1]))),
                2), 1)[:, np.newaxis]

            # trterm = np.array(
            #     [[-0.843010908566320], \
            #      [-0.582029313251303], \
            #      [-0.758961809411626], \
            #      [-0.534968553942584], \
            #      [-0.805424071511742], \
            #      [-0.206116671234491], \
            #      [-0.702439309633706], \
            #      [-0.704391419459464], \
            #      [-0.930137189649854], \
            #      [-0.916431159870648]])

            dHp = LossFunc(logP, lmb, lPred, zbel)
            dHy2 = np.mean(dHp, dtype=np.float64)

            ddHdx[d] = np.divide((dHy1 - dHy2), 2*e)
            if invertsign:
                ddHdx = -ddHdx
        # endfor
        return dH, ddHdx

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
        
        kbx = self.model.kernel.K(zb,x)
        kXx = self.model.kernel.K(self.model.X, x)
        kxx = self.model.likelihood.variance + self.model.kernel.K(x, x)
        # derivatives of kernel values 
        dkxx = self.model.kernel.gradients_X(kxx, x)
        dkxX = -1* self.model.kernel.gradients_X(np.ones((self.model.X.shape[0], x.shape[0])),self.model.X, x)
        dkxb = -1* self.model.kernel.gradients_X(np.ones((zb.shape[0], x.shape[0])), zb, x)
        # terms of innovation
        a = kxx - np.dot(kXx.T, (np.linalg.solve(cK, np.linalg.solve(cK.T, kXx))))
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
            mpm = None
            
            try:
                mpm     = MP * MP / P;
            except:
                print P, MP
                raise
            
            
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

    def predict_info_gain(self, entropy_fun, entropy_fun_p, zb, logP, X_lower, X_upper, Ne):
        # print logP

        # set random seed
        # np.random.seed(1)

        S0 = 0.5 * np.linalg.norm(X_upper - X_lower)
        D = X_lower.shape[0]
        mi = np.argmax(logP)
        xx = zb[mi,np.newaxis]
        Xstart = np.zeros((Ne, D))
        Xend = np.zeros((Ne, D))
        Xdhi = np.zeros((Ne, 1))
        Xdh = np.zeros((Ne,1))
        xxs = np.zeros((10*Ne, D))

        for i in range(1, 10*Ne):
            if i % 10 == 1 and i > 1:
                xx = X_lower + np.multiply(X_upper - X_lower, np.random.uniform(size=(1,D)))
            xx = self.slice_ShrinkRank_nolog(xx, entropy_fun_p, S0, True)
            xxs[i,:] = xx
            if i % 10 == 0:
                Xstart[(i/10)-1,:] = xx
                Xdhi[(i/10)-1],_ = entropy_fun(xx)

        search_cons = []
        for i in range(0, X_lower.shape[0]):
            xmin = X_lower[i]
            xmax = X_upper[i]
            search_cons.append({'type': 'ineq',
                                'fun' : lambda x: x - xmin})
            search_cons.append({'type': 'ineq',
                                'fun' : lambda x: xmax - x})
        search_cons = tuple(search_cons)
        minima = []
        for i in range(1, Ne):
            minima.append(scipy.optimize.minimize(
               fun=entropy_fun, x0=Xstart[i,np.newaxis], jac=True, method='slsqp', constraints=search_cons,
               options={'ftol':np.spacing(1), 'maxiter':20}
            ))

        print minima

    # This method corresponds to the function SampleBeliefLocations in the original ES code
    # It is assumed that the GP data structure is a Python dictionary
    # This function calls PI, EI etc and samples them (using their values)
    def sample_from_measure(self, xmin, xmax, n_representers, BestGuesses, acquisition_fn):

        # acquisition_fn = acquisition_fn(self.model)

        # If there are no prior observations, do uniform sampling
        if (self.model.X.size == 0):
            dim = xmax.size
            zb = np.add(np.multiply((xmax - xmin), np.random.uniform(size=(n_representers, dim))), xmin)
            # This is a rather ugly trick to get around the different ways of filling up an array from a sampled
            # distribution Matlab and NumPy use (by columns and rows respectively):
            zb = zb.flatten().reshape((dim, n_representers)).transpose()
            
            mb = np.dot(-np.log(np.prod(xmax - xmin)), np.ones((n_representers, 1)))
            return zb, mb

        # There are prior observations, i.e. it's not the first ES iteration
        dim = self.model.X.shape[1]

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

        restarts[0:(np.minimum(numblock, BestGuesses.shape[0])), ] = \
            BestGuesses[np.maximum(BestGuesses.shape[0]-numblock+1, 1) - 1:, ]

        restarts[(np.minimum(numblock, BestGuesses.shape[0])):numblock, ] = \
            np.add(xmin,
                   np.multiply((xmax-xmin),
                               np.random.uniform(
                                   size = (np.arange(np.minimum(numblock, BestGuesses.shape[0]) + 1, numblock + 1).size, dim)
                               )))

        xx = restarts[0,np.newaxis]
        subsample = 20
        for i in range(0, subsample * n_representers + 1): # Subasmpling by a factor of 10 improves mixing
            if (i % (subsample*10) == 0) and (i / (subsample*10.) < numblock):
                xx = restarts[i/(subsample*10), np.newaxis]
            xx = self.slice_ShrinkRank_nolog(xx, acquisition_fn, d0, True)
            if i % subsample == 0:
                zb[(i / subsample) - 1, ] = xx
                emb = acquisition_fn(xx)
                try:
            
                    mb[(i / subsample) - 1, 0]  = np.log(emb)
                except:
                    mb[(i / subsample) - 1, 0]  = -np.inf#sys.float_info.max
                    raise

        # Return values
        return zb, mb

    def projNullSpace(self, J, v):
        # Auxiliary function for the multivariate slice sampler
        if J.shape[1] > 0:
            return v - J.dot(J.transpose()).dot(v)
        else:
            return v

    def slice_ShrinkRank_nolog(self, xx, P, s0, transpose):
        # This function is equivalent to the similarly named function in the original ES code
        if transpose:
            xx = xx.transpose()

        # set random seed
        D = xx.shape[0]
        f = P(xx.transpose())
                
        
        try:
            logf = np.log(f)
        except:
            #print "~"*90
            logf = -np.inf#sys.float_info.max
        logy = np.log(np.random.uniform()) + logf
        

        theta = 0.95

        k = 0
        s = np.array([s0])
        c = np.zeros((D,0))
        J = np.zeros((D,0))
        while True:
            k += 1
            c = np.append(c, np.array(self.projNullSpace(J, xx + s[k-1] * np.random.randn(D,1))), axis = 1)
            sx = np.divide(1., np.sum(np.divide(1., s)))
            mx = np.dot(
                sx,
                np.sum(
                    np.multiply(
                        np.divide(1., s),
                        np.subtract(c, xx)
                    ),
                    1))
            xk = xx + self.projNullSpace(J, mx.reshape((D, 1)) + np.multiply(sx, np.random.normal(size=(D,1))))

            # TODO: add the derivative values (we're not considering them yet)
            # fk, dfk = P(xk.transpose())
            fk, dfk = P(xk.transpose(), derivative = True)
            
            try:
                logfk  = np.log(fk)
                dlogfk = np.divide(dfk, fk)
            except:
                logfk = - np.inf#sys.float_info.max
                dlogfk = 0
                 
            if (logfk > logy).all(): # accept these values
                xx = xk.transpose()
                return xx
            else: # shrink
                g = self.projNullSpace(J, dlogfk)
                if J.shape[1] < D - 1 and \
                   np.dot(g.transpose(), dlogfk) > 0.5 * np.linalg.norm(g) * np.linalg.norm(dlogfk):
                    J = np.append(J, np.divide(g, np.linalg.norm(g)), axis = 1)
                    # s[k] = s[k-1]
                    s = np.append(s, s[k-1])
                else:
                    s = np.append(s, np.multiply(theta, s[k-1]))
                    if s[k] < np.spacing(1):
                        print 'bug found: contracted down to zero step size, still not accepted.\n'
                    if transpose:
                        xx = xx.transpose()
                        return xx
                    else:
                        return xx





