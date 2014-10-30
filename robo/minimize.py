import DIRECT as _DIRECT

def _DIRECT_acquisition_fkt_wrapper(acq_f):
    def _l(x, user_data):
        return -acq_f(x), 0
    return _l
    
    
def DIRECT(acquisition_fkt, X_lower, X_upper):
    
    x, fmin, ierror = _DIRECT.solve(_DIRECT_acquisition_fkt_wrapper(acquisition_fkt), l=[X_lower], u=[X_upper], maxT=2000, maxf=2000)
    return x