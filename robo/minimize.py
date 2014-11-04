import sys
import StringIO
try:
    import DIRECT as _DIRECT
    def _DIRECT_acquisition_fkt_wrapper(acq_f):
        def _l(x, user_data):
            return -acq_f(x), 0
        return _l
    def DIRECT(acquisition_fkt, X_lower, X_upper):
        #stdout = sys.stdout
        #sys.stdout = StringIO.StringIO()
        x, fmin, ierror = _DIRECT.solve(_DIRECT_acquisition_fkt_wrapper(acquisition_fkt), l=[X_lower], u=[X_upper], maxT=2000, maxf=2000)
        #sys.stdout = stdout
        return x
    
except Exception, e:
    def DIRECT(acquisition_fkt, X_lower, X_upper):
        raise NotImplementedError("cannot find DIRECT library")
    
try:
    
    import cma as _cma
    def _cma_acquisition_fkt_wrapper(acq_f):
        def _l(*args, **kwargs):
            return -acq_f(*args, **kwargs)
        return _l
    def cma(acquisition_fkt, X_lower, X_upper):
        #stdout = sys.stdout
        #sys.stdout = StringIO.StringIO()
        x = _cma.fmin(_cma_acquisition_fkt_wrapper(acquisition_fkt), (X_upper + X_lower)*0.5, 2.5, options={"bounds":[X_lower, X_upper], "verbose":-1, "verb_log":sys.maxint})[0]
        #sys.stdout = stdout
        return x

except Exception, e:
    def cma(acquisition_fkt, X_lower, X_upper):
        raise NotImplementedError("cannot find cma library")


