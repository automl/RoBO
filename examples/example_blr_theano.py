import numpy as np

import theano
import theano.tensor as T
import theano.tensor.nlinalg

from scipy import optimize




class BayesianLinearRegression(object):

	def __init__(self, alpha=1, beta=1000, basis_func=None,
				 prior=None, do_mcmc=True, n_hypers=20, chain_length=2000,
				 burnin_steps=2000, rng=None):
		"""
		Implementation of Bayesian linear regression. See chapter 3.3 of the book
		"Pattern Recognition and Machine Learning" by Bishop for more details.
		Parameters
		----------
		alpha: float
			Specifies the variance of the prior for the weights w
		beta : float
			Defines the inverse of the noise, i.e. beta = 1 / sigma^2
		basis_func : function
			Function handle to transfer the input with via basis functions
			(see the code above for an example)
		prior: Prior object
			Prior for alpa and beta. If set to None the default prior is used
		do_mcmc: bool
			If set to true different values for alpha and beta are sampled via MCMC from the marginal log likelihood
			Otherwise the marginal log likehood is optimized with scipy fmin function
		n_hypers : int
			Number of samples for alpha and beta
		chain_length : int
			The chain length of the MCMC sampler
		burnin_steps: int
			The number of burnin steps before the sampling procedure starts
		rng: np.random.RandomState
			Random number generator
		"""

		if rng is None:
			self.rng = np.random.RandomState(np.random.randint(0, 10000))
		else:
			self.rng = rng


		self.alpha = alpha
		self.beta = beta
		self.basis_func = basis_func
		self.prior = prior
		self.do_mcmc = do_mcmc
		self.n_hypers = n_hypers
		self.chain_length = chain_length
		self.burned = False
		self.burnin_steps = burnin_steps
		self.models = None

		theta = T.dvector('theta')
		self.Phi = theano.shared('Phi')
		self.y = T.dmatrix('y')
		
		self.mll = theano.function([theta], self.marginal_log_likelihood_theano(theta),on_unused_input='warn')
		

	def marginal_log_likelihood(self, theta):
		"""
		Log likelihood of the data marginalised over the weights w. See chapter 3.5 of
		the book by Bishop of an derivation.
		Parameters
		----------
		theta: np.array(2,)
			The hyperparameter alpha and beta on a log scale
		Returns
		-------
		float
			lnlikelihood + prior
		"""
		return(self.mll(theta))


	def marginal_log_likelihood_theano(self, theta):
		"""
		Log likelihood of the data marginalised over the weights w. See chapter 3.5 of
		the book by Bishop of an derivation.
		Parameters
		----------
		theta: np.array(2,)
			The hyperparameter alpha and beta on a log scale
		Returns
		-------
		float
			lnlikelihood + prior
		"""
		
		# Theta is on a logscale        
		t = T.exp(theta)
		alpha = t[0]
		beta = t[1]

		N = T.shape(X)[0]	# number of data points
		M = T.shape(X)[1]	# number of features

		A =  beta  * T.dot(self.Phi.T, X)
		A += alpha * T.identity_like(A)

		A_inv = T.inv(A)

		m = beta * T.dot(T.dot(A, self.Phi.T), y)

		mll = M / 2 * T.log(alpha)
		mll += N / 2 * np.log(beta)
		
		mll -= beta / 2. * (y - T.dot(X, m)).norm(2)
		mll -= alpha / 2. * T.dot(m.T, m)

		mll -= 0.5 * T.log(T.nlinalg.det(A))


		if self.prior is not None:
			mll += self.prior.lnprob(theta)

		return(mll)


	def train(self, X, y, do_optimize=True):
		"""
		First optimized the hyperparameters if do_optimize is True and then computes
		the posterior distribution of the weights. See chapter 3.3 of the book by Bishop
		for more details.
		Parameters
		----------
		X: np.ndarray (N, D)
			Input data points. The dimensionality of X is (N, D),
			with N as the number of points and D is the number of features.
		y: np.ndarray (N,)
			The corresponding target values.
		do_optimize: boolean
			If set to true the hyperparameters are optimized otherwise
			the default hyperparameters are used.
		"""

		self.X = X

		if self.basis_func is not None:
			self.Phi = self.basis_func(X)
		else:
			self.Phi = self.X

		self.y = y

		if do_optimize:
			if self.do_mcmc:
				raise NotImplementedError("TODO")
			else:
				# Optimize hyperparameters of the Bayesian linear regression        
				res = optimize.fmin(self.negative_mll, self.rng.rand(2))
				self.hypers = [[np.exp(res[0]), np.exp(res[1])]]
				
		else:
			self.hypers = [[self.alpha, self.beta]]
		
		self.models = []
		for sample in self.hypers:
			alpha = sample[0]
			beta = sample[1]

			logger.debug("Alpha=%f ; Beta=%f" % (alpha, beta))

			S_inv  = beta * np.dot(self.Phi.T, self.Phi)
			S_inv += alpha* np.eye(self.Phi.shape[1])

			S = np.linalg.inv(S_inv)
			m = beta * np.dot(np.dot(S, self.Phi.T), self.y)

			self.models.append((m, S))


N = 20
x = np.linspace(0,1,N)

def quadratic_basis_func(x):
	x = np.append(x**2, x, axis=1)
	return np.append(x, np.ones([x.shape[0], 1]), axis=1)


X = quadratic_basis_func(x[:,None])
y = 0.5*x + 0.1




from IPython import embed

embed()
