import emcee
import numpy as np

from scipy import optimize
from scipy.stats import multivariate_normal

from robo.models.base_model import BaseModel


class BayesianLinearRegression(BaseModel):

    def __init__(self, alpha=1, beta=1000, basis_func=None,
                 prior=None, do_mcmc=True, n_hypers=20, chain_length=2000,
                 burnin_steps=2000):

        self.X = None
        self.y = None
        self.alpha = alpha
        self.beta = beta
        self.basis_func = basis_func
        self.prior = prior
        self.do_mcmc = do_mcmc
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps        

    def marginal_log_likelihood(self, theta):
        # Theta is on a log scale
        alpha = np.exp(theta[0])
        beta = np.exp(theta[1])

        D = self.X.shape[1]
        N = self.X.shape[0]

        K = beta * np.dot(self.X.T, self.X)
        K += np.eye(self.X.shape[1]) * alpha
        K_inv = np.linalg.inv(K)
        m = beta * np.dot(K_inv, self.X.T)
        m = np.dot(m, self.y)

        mll = D / 2 * np.log(alpha)
        mll += N / 2 * np.log(beta)
        mll -= N / 2 * np.log(2 * np.pi)
        mll -= beta / 2. * np.linalg.norm(self.y - np.dot(self.X, m), 2)
        mll -= alpha / 2. * np.dot(m.T, m)
        mll -= 0.5 * np.log(np.linalg.det(K))

        if self.prior is not None:
            mll += self.prior.lnprob(theta)

        return mll

    def negative_mll(self, theta):
        negative_mll = -self.marginal_log_likelihood(theta)
        return negative_mll

    @BaseModel._check_shapes_train
    def train(self, X, y, do_optimize=True):
        if self.basis_func is not None:
            self.X = self.basis_func(X)
        else:
            self.X = X
        self.y = y

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
    
                # Save the current position, it will be the start point in
                # the next iteration
                self.p0 = pos
    
                # Take the last samples from each walker
                self.hypers = np.exp(self.sampler.chain[:, -1])
            else:
                # Optimize hyperparameters of the Bayesian linear regression        
                res = optimize.fmin(self.negative_mll, np.random.rand(2))
                self.hypers = [[np.exp(res[0]), np.exp(res[1])]]
                
        else:
            self.hypers = [[self.alpha, self.beta]]
        
        self.K_inv = []
        self.models = []
        for sample in self.hypers:
            alpha = sample[0]            
            beta = sample[1]
            
            K = beta * np.dot(self.X.T, self.X)
            K += np.eye(self.X.shape[1]) * alpha

            K_inv = np.linalg.inv(K)
            m = beta * np.dot(np.dot(K_inv, self.X.T), self.y)
            print(K_inv.shape)
            self.K_inv.append(K_inv)
            self.models.append(m)

    def posterior(self, x):
        mean = self.beta * np.dot(np.dot(self.K_inv, self.X.T), self.y)
        
        p = multivariate_normal.pdf(x, mean=mean[:, 0], cov=self.K_inv)
        
        return p

    @BaseModel._check_shapes_predict
    def predict(self, X_test):
        if self.basis_func is not None:
            X_transformed = self.basis_func(X_test)
        else:
            X_transformed = X_test
            
        # Marginalise predictions over hyperparameters
        mu = np.zeros([len(self.hypers), X_transformed.shape[0]])
        var = np.zeros([len(self.hypers), X_transformed.shape[0]])
    
        for i, h in enumerate(self.hypers):
            mu[i] = np.dot(X_transformed, self.models[i])
            var[i] = np.diag(np.dot(np.dot(X_transformed, self.K_inv[i]), X_transformed.T)) + 1. / h[1]

        # See the algorithm runtime prediction paper by Hutter et al
        # for the derivation of the total variance
        m = mu.mean(axis=0)
        v = np.mean(mu ** 2 + var) - m ** 2

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        return m, v
