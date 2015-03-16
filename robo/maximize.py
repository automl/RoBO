import sys
import StringIO
import numpy as np
import scipy
import emcee
from sampling import slice_ShrinkRank_nolog
try:
    import DIRECT as _DIRECT
    def _DIRECT_acquisition_fkt_wrapper(acq_f):
        def _l(x, user_data):
            return -acq_f(np.array([x])), 0
        return _l
    def DIRECT(acquisition_fkt, X_lower, X_upper):
        #stdout = sys.stdout
        #sys.stdout = StringIO.StringIO()
        x, fmin, ierror = _DIRECT.solve(_DIRECT_acquisition_fkt_wrapper(acquisition_fkt), l=[X_lower], u=[X_upper], maxT=2000, maxf=2000)
        #sys.stdout = stdout
        return np.array([x])
    
except Exception, e:
    def DIRECT(acquisition_fkt, X_lower, X_upper):
        raise NotImplementedError("cannot find DIRECT library")

try:
    import cma as _cma
    def _cma_fkt_wrapper(acq_f, derivative=False):
       def _l(x, *args, **kwargs):
           x = np.array([x])
           return -acq_f(x, derivative=derivative,*args, **kwargs)
       return _l
    def cma(acquisition_fkt, X_lower, X_upper):
        #stdout = sys.stdout
        #sys.stdout = StringIO.StringIO()
        x = _cma.fmin(_cma_fkt_wrapper(acquisition_fkt), (X_upper + X_lower)*0.5, 0.6, options={"bounds":[X_lower, X_upper], "verbose":-1, "verb_log":sys.maxint})[0]
        #sys.stdout = stdout
        return np.array([x])

except Exception, e:
    def cma(acquisition_fkt, X_lower, X_upper):
        raise NotImplementedError("cannot find cma library")

def grid_search(acquisition_fkt, X_lower, X_upper, resolution=1000):
    from numpy import linspace, array
    if  X_lower.shape[0] >1 :
        raise RuntimeError("grid search works for 1D only")
    x = linspace(X_lower[0], X_upper[0], resolution).reshape((resolution, 1, 1))
    # y = array(map(acquisition_fkt, x))
    ys = [None] * resolution
    for i in range(resolution):
        ys[i] = acquisition_fkt(x[i])
    y = array(ys)
    x_star = x[y.argmax()]
    return x_star

def _sample_optimizer_fkt_wrapper(acq_f):
    def _l(x, *args, **kwargs):
        a = [-1* x for x in acq_f(x, derivative=True,*args, **kwargs)]
        return a
    return _l

def _scipy_optimizer_fkt_wrapper(acq_f):
    def _l(x, *args, **kwargs):
        x = np.array([x])
        a = acq_f(x, derivative=True,*args, **kwargs)
        return -a[0], -a[1]
    return _l

def sample_optimizer(acquisition_fkt, X_lower, X_upper, Ne=20):
    if hasattr(acquisition_fkt, "_get_most_probable_minimum"):
        xx = acquisition_fkt._get_most_probable_minimum()
    else:
        xx = np.add(np.multiply((X_lower - X_upper), np.random.uniform(size=(1, X_lower.shape[0]))), X_lower)
    fun_p = acquisition_fkt
    fun = _sample_optimizer_fkt_wrapper(acquisition_fkt)
    sc_fun = _scipy_optimizer_fkt_wrapper(acquisition_fkt)
    S0 = 0.5 * np.linalg.norm(X_upper - X_lower)
    D = X_lower.shape[0]
    Xstart = np.zeros((Ne, D))
    Xend = np.zeros((Ne, D))
    Xdhi = np.zeros((Ne, 1))
    Xdh = np.zeros((Ne,1))
    xxs = np.zeros((10*Ne, D))
    for i in range(1, 10*Ne):
        if i % 10 == 1 and i > 1:
            xx = X_lower + np.multiply(X_upper - X_lower, np.random.uniform(size=(1,D)))
        xx = slice_ShrinkRank_nolog(xx, fun_p, S0, True)
        xxs[i,:] = xx
        if i % 10 == 0:
            Xstart[(i/10)-1,:] = xx
            Xdhi[(i/10)-1],_ = fun(xx)
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
           fun=sc_fun, x0=Xstart[i,np.newaxis], jac=True, method='slsqp', constraints=search_cons,
           options={'ftol':np.spacing(1), 'maxiter':20}
        ))
    # X points:
    Xend = np.array([res.x for res in minima])
    # Objective function values:
    Xdh = np.array([res.fun for res in minima])
    new_x = Xend[np.argmin(Xdh)]
    return [new_x]
