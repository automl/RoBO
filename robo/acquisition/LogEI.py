from scipy.stats import norm
import numpy as np
from robo import BayesianOptimizationError
from robo.acquisition.base import AcquisitionFunction 
class LogEI(AcquisitionFunction):
    long_name = "Logarithm  of Expected Improvement"
    def __init__(self, model,  X_lower, X_upper, par = 0.01, **kwargs):
        self.model = model
        self.par = par
        self.X_lower = X_lower
        self.X_upper = X_upper
        
    def __call__(self,X,  Z=None, derivative=False, **kwargs):
        if derivative:
            raise BayesianOptimizationError(BayesianOptimizationError.NO_DERIVATIVE,
                                            "LogEI does not support derivative calculation until now")
        
        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            #print "return -inf"
            return np.array([[- np.finfo(np.float).max]])
        m, v = self.model.predict(X, Z)
        
        eta = self.model.getCurrentBest()
        s = np.sqrt(v)
        z = (eta - m) / s - self.par
        log_ei = np.zeros((m.size, 1))
        for i in range(0, m.size):
            mu, sigma = m[i], s[i]
            par_s = self.par * sigma
            # Degenerate case 1: first term vanishes
            if abs(eta - mu - par_s) == 0:
                if sigma > 0:
                    log_ei[i] = np.log(sigma) + norm.logpdf(z[i])
                else:
                    log_ei[i] = -float('Inf')
            # Degenerate case 2: second term vanishes and first term has a special form.
            elif sigma == 0:
                if mu < eta - par_s:
                    log_ei[i] = np.log(eta - mu - par_s)
                else:
                    log_ei[i] = -float('Inf')
            # Normal case
            else:
                b = np.log(sigma) + norm.logpdf(z[i])
                # log(y+z) is tricky, we distinguish two cases:
                if eta - par_s > mu:
                    # When y>0, z>0, we define a=ln(y), b=ln(z).
                    # Then y+z = exp[ max(a,b) + ln(1 + exp(-|b-a|)) ],
                    # and thus log(y+z) = max(a,b) + ln(1 + exp(-|b-a|))
                    a = np.log(eta - mu - par_s) + norm.logcdf(z[i])
                    log_ei[i] = max(a, b) + np.log(1 + np.exp(-abs(b-a)))
                else:
                    # When y<0, z>0, we define a=ln(-y), b=ln(z), and it has to be true that b >= a in order to satisfy y+z>=0.
                    # Then y+z = exp[ b + ln(exp(b-a) -1) ],
                    # and thus log(y+z) = a + ln(exp(b-a) -1)
                    a = np.log(mu - eta + par_s) + norm.logcdf(z[i])
                    if a >= b:
                        # a>b can only happen due to numerical inaccuracies or approximation errors
                        log_ei[i] = -np.Infinity
                    else:
                        log_ei[i] = b + np.log(1 - np.exp(a - b))
        return log_ei

