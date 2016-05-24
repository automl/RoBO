
import numpy as np
import emcee

from robo.models.base_model import BaseModel
from scipy import optimize


class BayesianLinearRegression(BaseModel):

    def __init__(self, alpha=1, beta=1000, basis_func=None,
                 prior=None, do_mcmc=True,  n_hypers=20, chain_length=2000,
                 burnin_steps=2000, *args, **kwargs):

        self.X = None
        self.y = None
        self.alpha = alpha
        self.beta = beta
        self.basis_func = basis_func
        self.prior = prior
        self.do_mcmc = do_mcmc
        self.do_mcmc = do_mcmc
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps        

    def marginal_log_likelihood(self, theta):
        # Theta is on a logscale        
        alpha = np.exp(theta[0])
        beta = np.exp(theta[1])

        D = self.X.shape[1]
        N = self.X.shape[0]

        K = beta * np.dot(self.X.T, self.X)
        K += np.eye(self.X.shape[1]) * alpha
        K_inv = np.linalg.inv(K)
        m = beta * np.dot(K_inv, self.X.T)
        m = np.dot(m, self.Y)

        mll = D / 2 * np.log(alpha)
        mll += N / 2 * np.log(beta)
        mll -= N / 2 * np.log(2 * np.pi)
        mll -= beta / 2. * np.linalg.norm(self.Y - np.dot(self.X, m), 2)
        mll -= alpha / 2. * np.dot(m.T, m)
        mll -= 0.5 * np.log(np.linalg.det(K))

        if self.prior is not None:
            mll += self.prior.lnprob(theta)

        return mll[0, 0]

    def nmll(self, theta):        
        
        nmll = -self.marginal_log_likelihood(theta)
        
        return nmll

    def train(self, X, Y, do_optimize=True):
        if self.basis_func is not None:
            self.X = self.basis_func(X)
        else:
            self.X = X
        self.Y = Y

        if do_optimize:
            if self.do_mcmc:
                self.sampler = emcee.EnsembleSampler(self.n_hypers,
                                                     self.X.shape[1],
                                                 self.marginal_log_likelihood)

                # Do a burn-in in the first iteration
                if not self.burned:
                    # Initialize the walkers by sampling from the prior
                    self.p0 = self.prior.sample_from_prior(self.n_hypers)
                    # Run MCMC sampling
                    self.p0, _, _ = self.sampler.run_mcmc(self.p0,
                                                          self.burnin_steps)
    
                    self.burned = True
    
                # Start sampling
                pos, _, _ = self.sampler.run_mcmc(self.p0,
                                                  self.chain_length)
    
                # Save the current position, it will be the startpoint in
                # the next iteration
                self.p0 = pos
    
                # Take the last samples from each walker
                self.hypers = np.exp(self.sampler.chain[:, -1])
            else:
                # Optimize hyperparameters of the Bayesian linear regression        
                res = optimize.fmin(self.nmll, np.random.rand(2))
                self.hypers = [[np.exp(res[0]), np.exp(res[1])]]
                
        else:
            self.hypers = [[self.alpha, self.beta]]
        
        self.K_inv = []
        self.m = []
        for sample in self.hypers:
            alpha = sample[0]            
            beta = sample[1]
            
            K = beta  * np.dot(self.X.T, self.X)        
            K +=  np.eye(self.X.shape[1]) * alpha

            K_inv = np.linalg.inv(K)
            m = beta * np.dot(np.dot(K_inv, self.X.T), self.Y)

            self.K_inv.append(K_inv)
            self.m.append(m)

    def posterior(self, x):
        from scipy.stats import multivariate_normal
        mean = self.beta * np.dot(np.dot(self.K_inv, self.X.T), self.Y)
        
        p = multivariate_normal.pdf(x, mean=mean[:, 0], cov=self.K_inv)
        
        return p

    def predict(self, X):
        if self.basis_func is not None:
            X_ = self.basis_func(X)
        else:
            X_ = X
            
                    # Marginalise predictions over hyperparameters of the BLR
        mu = np.zeros([len(self.hypers), X_.shape[1]])
        var = np.zeros([len(self.hypers), X_.shape[1]])
    
        for i, h in enumerate(self.hypers):

            mu[i] =  np.dot(X_, self.m[i])
            var[i] = np.dot(np.dot(X_, self.K_inv[i]), X_.T) + 1. / h[1]

        # See the algorithm runtime prediction paper by Hutter et al
        # for the derivation of the total variance
        m = np.array([[mu.mean()]])
        v = np.mean(mu ** 2 + var) - m ** 2

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v[np.diag_indices(v.shape[0])] = \
                    np.clip(v[np.diag_indices(v.shape[0])],
                            np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0


        return m, v


    def predictive_gradients(self, Xnew, X=None):
        """
        Calculates the predictive gradients (gradient of the prediction)
        :param Xnew: The points to predict the gradient for
        :param X: TODO: Not implemented yet
        :return: Gradients(?)
        """
        raise NotImplementedError()
