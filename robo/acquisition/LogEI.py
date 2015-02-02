from scipy.stats import norm
import numpy as np
class LogEI(object):
    def __init__(self, model, par = 0.01, **kwargs):
        self.model = model
        self.par = par,
    def __call__(self, x,  Z=None, **kwargs):
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
                # log(y+z) is tricky, we distinguish two cases:
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

    def update(self, model):
        self.model = model
