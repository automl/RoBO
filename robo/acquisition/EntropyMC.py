import sys
from scipy.stats import norm
import scipy
import numpy as np
import emcee
from robo.loss_functions import logLoss
from robo import BayesianOptimizationError
from robo.sampling import sample_from_measure, montecarlo_sampler
from robo.acquisition.LogEI import LogEI
from robo.acquisition.base import AcquisitionFunction 
from robo.acquisition import Entropy
sq2 = np.sqrt(2)
l2p = np.log(2) + np.log(np.pi)
eps = np.finfo(np.float32).eps

class EntropyMC(Entropy):
    def __init__(self, model, X_lower, X_upper, Nb = 50, Nf= 200, sampling_acquisition = None, sampling_acquisition_kw = {"par":2.4}, Np=15, loss_function=None, **kwargs):
        self.model = model
        self.Nb = Nb
        self.Nf = Nf
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
    
    def __call__(self, X, Z=None, **kwargs):
        return self.dh_fun(X)
    
    def update(self, model):
        self.model = model
        self.sampling_acquisition.update(model)
        self.update_representer_points()
        self.W = np.random.randn(1, self.Np)
        self.Mb, self.Vb = self.model.predict(self.zb, full_cov=True) 
        self.F = np.random.multivariate_normal(mean=np.zeros(self.Nb), cov=np.eye(self.Nb), size=self.Nf)
        self.cVb = np.linalg.cholesky(self.Vb)
        self.f = np.add(np.dot(self.cVb, self.F.T).T, self.Mb).T
        self.pmin = self.calc_pmin(self.f)
        self.logP = np.log(self.pmin)
    
    def calc_pmin(self, f):
        if len(f.shape) == 3:
            f = f.reshape(f.shape[0],f.shape[1]*f.shape[2])
        mins = np.argmin(f, axis=0)
        c = np.bincount(mins)
        min_count = np.zeros((self.Nb,))
        min_count[:len(c)] += c
        pmin = (min_count/f.shape[1])[:,None]
        pmin[np.where(pmin<1e-70)] = 1e-70
        return pmin

    def change_pmin_by_innovation(self, x, f):
        Lx, _ = self._gp_innovation_local(x)
        dMdb = Lx
        dVdb = -Lx.dot(Lx.T)
        stoch_changes = dMdb.dot(self.W)
        Mb_new = self.Mb[:,None] + stoch_changes
        Vb_new = self.Vb + dVdb
        cVb_new = np.linalg.cholesky(Vb_new)
        f_new = np.dot(cVb_new, self.F.T)
        f_new = f_new[:,:,None]
        Mb_new = Mb_new[:,None, :]
        f_new = Mb_new + f_new
        return self.calc_pmin(f_new)
        
    def dh_fun(self, x):
        # TODO: should this be shape[1] ?
        if x.shape[0] > 1:
            raise BayesianOptimizationError(BayesianOptimizationError.SINGLE_INPUT_ONLY, "dHdx_local is only for single x inputs")
        new_pmin  = self.change_pmin_by_innovation(x, self.f)
        # Calculate the Kullback-Leibler divergence w.r.t. this pmin approximation
        H_old = np.sum(np.multiply(self.pmin, (self.logP + self.lmb)))
        H_new = np.sum(np.multiply(new_pmin, (np.log(new_pmin) + self.lmb)))
        return np.array([[ - H_new + H_old]])
    
    def plot(self, fig, minx, maxx, plot_attr={"color":"red"}, resolution=1000):
        n = len(fig.axes)
        for i in range(n):
            fig.axes[i].change_geometry(n+3, 1, i+1) 
        ax = fig.add_subplot(n+3, 1, n+1)
        bar_ax = fig.add_subplot(n+3, 1, n+2)
        other_acq_ax = fig.add_subplot(n+3, 1, n+3)
        plotting_range = np.linspace(minx, maxx, num=resolution)
        acq_v =  np.array([ self(np.array([x]), derivative=True)[0][0] for x in plotting_range[:,np.newaxis] ])
        ax.plot(plotting_range, acq_v, **plot_attr)
        zb = self.zb
        bar_ax.plot(zb, np.zeros_like(zb), "g^")
        ax.set_xlim(minx, maxx)
        bar_ax.bar(zb, self.pmin[:,0], width=(maxx - minx)/200, color="yellow")
        bar_ax.set_xlim(minx, maxx)
        other_acq_ax.plot(zb, self.f[:,0], "g+")
        ss = np.empty_like(zb)
        ss.fill(0.2)
        bar_ax.plot(self.zb[:,0], ss, "r." , markeredgewidth=5.0)
        self.change_pmin_by_innovation(np.array([[0.4]]), self.f)
        #other_acq_ax.set_xlim(minx, maxx)
        #self.sampling_acquisition.plot(fig, minx, maxx, plot_attr={"color":"orange"})#, logscale=True)
        ax.set_title(str(self))
        return ax