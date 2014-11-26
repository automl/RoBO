"""
this module contains acquisition functions that have high values
where the objective function is low.

.. class:: AcquisitionFunc

    An acquisition function is a class that gets instatiated with a model 
    and optional additional parameters. It then gets called via a maximizer.

    .. method:: __init__(model, **optional_kwargs)
                
        :param model: A model should have at least the function getCurrentBest() 
                      and predict(X, Z)

    .. method:: __call__(X, Z=None)
               
        :param X: X values, where to evaluate the acquisition function 
        :param Z: instance features to evaluate at. Could be None.
"""
from scipy.stats import norm
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

class Entropy(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, X, Z=None, **kwargs):
        raise NotImplementedError
    def model_changed(self):
        raise NotImplementedError

    # This method corresponds to the function SampleBeliefLocations in the original ES code
    # It is assumed that the GP data structure is a Python dictionary
    # This function calls PI, EI etc and samples them (using their values)
    def sample_from_measure(self, xmin, xmax, n_representers, BestGuesses, acquisition_fn):
        # TODO: does it make sense to use the same GP model for the acquisition function used in the sampling of representer points?
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

        xx = restarts[0,np.newaxis]
        subsample = 20 # why this value?
        for i in range(0, subsample * n_representers + 1): # Subasmpling by a factor of 10 improves mixing (?)
            # print i,
            if (i % (subsample*10) == 0) & (i / (subsample*10.) < numblock):
                xx = restarts[i/(subsample*10), np.newaxis]
                # print str(xx)
            xx = self.slice_ShrinkRank_nolog(xx, acquisition_fn, d0, True)
            if i % subsample == 0:
                zb[(i / subsample) - 1, ] = xx
                emb, _ = acquisition_fn(xx)
                mb[(i / subsample) - 1, 0]  = np.log(emb)

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

        D = xx.shape[0]
        f, _ = P(xx.transpose())
        logf = np.log(f)
        logy = np.log(np.random.uniform()) + logf

        theta = 0.95

        k = 0
        s = np.array([s0])
        # print '*'*30
        # print s.shape
        c = np.zeros((D,0))
        J = np.zeros((0,0))

        while True:
            k += 1
            # print '*'*30
            # print s
            c = np.append(c, np.array(self.projNullSpace(J, xx + s[k-1] * np.random.normal(size=(D,1)))), axis = 1)
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

            # TODO: add the derivative values (we're not considering them yet)
            # fk, dfk = P(xk.transpose())
            fk, dfk = P(xk.transpose())
            logfk  = np.log(fk)
            dlogfk = np.divide(dfk, fk)

            if logfk > logy: # accept these values
                xx = xk.transpose()
                return xx
            else: # shrink
                g = self.projNullSpace(J, dlogfk)
                if J.shape[1] < D - 1 and \
                   np.dot(g.transpose(), dlogfk) > 0.5 * np.linalg.norm(g) * np.linalg.norm(dlogfk):
                    J = np.append(J, np.divide(g, np.linalg.norm(g)))
                    s[k] = s[k-1]
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






class EI(object):
    def __init__(self, model, par = 0.01, **kwargs):
        self.model = model
        self.par = par
        if len(self.model.X) > 0:
            # alpha = GP.cK \ (GP.cK' \ GP.y);
            self.alpha = np.linalg.solve(self.model.cK, np.linalg.solve(self.model.cK.transpose(), self.model.Y))
            # print "alpha: ", self.alpha
    def __call__(self, x, Z=None, **kwargs):

        dim = x.shape[1]
        f_est = self.model.predict(x)
        # print "f_est: ", f_est
        eta = self.model.getCurrentBest()
        z = (eta - f_est[0] + self.par) / f_est[1]
        f = (eta - f_est[0] + self.par) * norm.cdf(z) + f_est[1] * norm.pdf(z)

        # Derivative values:
        # Derivative of kernel values:
        dkxX = self.model.kernel.dK_dX(np.array([np.ones(len(self.model.X))]), self.model.X, x)
        # print "dkxX: ", dkxX
        dkxx = self.model.kernel.dK_dX(np.array([np.ones(len(self.model.X))]), self.model.X)
        # print "dkxx: ", dkxx

        # dm = derivative of the gaussian process mean function
        dmdx = np.dot(dkxX.transpose(), self.alpha)
        # print "dmdx: ", dmdx
        # ds = derivative of the gaussian process covariance function
        dsdx = np.zeros((dim, 1))
        # print "dim: ", dim
        # print self.model.K[0,None]
        # print self.model.K[0,None].shape
        for i in range(0, dim):
            dsdx[i] = np.dot(0.5 / f_est[1], dkxx[0,dim-1] - 2 * np.dot(dkxX[:,dim-1].transpose(),
                                                                        np.linalg.solve(self.model.cK,
                                                                                        np.linalg.solve(self.model.cK.transpose(),
                                                                                                        self.model.K[0,None].transpose()))))

        # print "dmdx: ", dmdx
        # print "dsdx: ", dsdx
        df = -dmdx * norm.cdf(z) + dsdx * norm.pdf(z)
        # print df
        return f, df

    def model_changed(self):
        pass


class LogEI(object):
    def __init__(self, model, par = 0.01):
        self.model = model
        self.par = par,
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

    def model_changed(self):
        pass
