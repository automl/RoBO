import sys
import StringIO
import numpy as np
try:
    import DIRECT as _DIRECT
    def _DIRECT_acquisition_fkt_wrapper(acq_f):
        def _l(x, user_data):
            return -acq_f(x), 0
        return _l
    def DIRECT(acquisition_fkt, X_lower, X_upper):
        stdout = sys.stdout
        sys.stdout = StringIO.StringIO()
        x, fmin, ierror = _DIRECT.solve(_DIRECT_acquisition_fkt_wrapper(acquisition_fkt), l=[X_lower], u=[X_upper], maxT=2000, maxf=2000)
        sys.stdout = stdout
        return np.array([x])
    
except Exception, e:
    def DIRECT(acquisition_fkt, X_lower, X_upper):
        raise NotImplementedError("cannot find DIRECT library")
    
try:
    
    import cma as _cma
    def _cma_acquisition_fkt_wrapper(acq_f):
        def _l(x, *args, **kwargs):
            x = np.array([x])
            return -acq_f(x,*args, **kwargs)
        return _l
    def cma(acquisition_fkt, X_lower, X_upper):
        #stdout = sys.stdout
        #sys.stdout = StringIO.StringIO()
        x = _cma.fmin(_cma_acquisition_fkt_wrapper(acquisition_fkt), (X_upper + X_lower)*0.5, 0.6, options={"bounds":[X_lower, X_upper], "verbose":-1, "verb_log":sys.maxint})[0]
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
    
    y = array(map(acquisition_fkt, x))
    x_star = x[y.argmax()]
    return x_star

def predict_info_gain(fun, fun_p, zb, logP, X_lower, X_upper, Ne):
    import scipy.optimize.minimize
    from sampling import slice_ShrinkRank_nolog

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
           fun=fun, x0=Xstart[i,np.newaxis], jac=True, method='slsqp', constraints=search_cons,
           options={'ftol':np.spacing(1), 'maxiter':20}
        ))

    # X points:
    Xend = np.array([res.x for res in minima])
    # Objective function values:
    Xdh = np.array([res.fun for res in minima])

    # print "xend: ", Xend
    # print "xdh: ", Xdh
    # print "the desired value is then: ", Xend[np.argmin(Xdh)]
    return Xend[np.argmin(Xdh)]
