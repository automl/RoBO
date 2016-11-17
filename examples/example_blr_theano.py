import logging


import theano
import theano.tensor as T
import theano.tensor.nlinalg as nlinalg

import matplotlib.pyplot as plt


import numpy as np
from scipy import optimize
from IPython import embed

import sys
sys.path.append("/ihome/sfalkner/repositories/github/RoBO/")

from robo.util.hmc import HMC_sampler



logger = logging.getLogger(__name__)



def quadratic_basis_func(x):
	x = np.append(x**2, x, axis=1)
	return np.append(x, np.ones([x.shape[0], 1]), axis=1)

def linear_basis_func(x):
	return np.append(x, np.ones([x.shape[0], 1]), axis=1)






# Some Theano functions to get gradients later

def static_compute_blr_matrices(Phi, y, alpha, beta):

	A = beta * T.dot(Phi.T, Phi)
	A += T.eye(T.shape(Phi)[1]) * alpha
	A_inv = nlinalg.matrix_inverse(A)
	m = beta * T.dot(A_inv, Phi.T)
	m = T.dot(m, y)

	return (m, A_inv)


def static_marginal_log_likelihood(Phi, y, theta):
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

	# Theta is on a log scale
	alpha = T.exp(theta[0])
	beta = T.exp(theta[1])

	D = Phi.shape[1]
	N = Phi.shape[0]

	A = beta * T.dot(Phi.T, Phi)
	A += T.eye(T.shape(Phi)[1]) * alpha
	A_inv = nlinalg.matrix_inverse(A + 1e-7*T.identity_like(A))
	m = beta * T.dot(A_inv, Phi.T)
	m = T.dot(m, y)

	mll = D / 2 * T.log(alpha)
	mll += N / 2 * T.log(beta)
	
	mll -= beta / 2. * T.sum(T.power(y - T.dot(Phi, m), 2))
	mll -= alpha / 2. * T.dot(m.T, m)
	mll -= 0.5 * T.log(nlinalg.det(A))

	return mll


def batched_marginal_log_likelihood(Phi, y, thetas):
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

	# Theta is on a log scale
	alpha = T.exp(thetas[:,0])[:,np.newaxis, np.newaxis]
	beta = T.exp(thetas[:,1])[:,np.newaxis, np.newaxis]

	D = Phi.shape[1]
	N = Phi.shape[0]
	B = beta.shape[0]

	A = beta * T.dot(Phi.T, Phi)
	A += T.eye(T.shape(Phi)[1]) * alpha
	A_inv,_ = theano.scan( lambda Ai: nlinalg.matrix_inverse(Ai + 1e-7*T.identity_like(Ai)), sequences=A)
	m_, _ = theano.scan( lambda bi, Aii: bi*T.dot(Aii, Phi.T), sequences=[beta, A_inv])
	m, _ = theano.scan( lambda mi: T.dot(mi, y), sequences=m_)

	mll = D / 2 * T.log(alpha[:,0,0])
	mll += N / 2 * T.log(beta[:,0,0])

	mll -= beta[:,0,0] / 2. * T.sum(T.power(y[:,np.newaxis]-T.dot(Phi, m.T), 2), axis=0)
	mll -= alpha[:,0,0]/ 2. * (m*m).sum(axis=1)


	logdets,_ = theano.scan( lambda Ai: T.log(nlinalg.det(Ai)), sequences=A)

	mll -= 0.5 * logdets
	return mll





Phi1 = T.dmatrix('Phi1')
y1 = T.dvector('y1')
theta1 = T.dvector('theta1')

marginal_log_likelihood_theano = theano.function([Phi1, y1, theta1], static_marginal_log_likelihood(Phi1,y1,theta1))


Phi2 = T.dmatrix('Phi2')
y2 = T.dvector('y2')
theta2 = T.dvector('theta2')


gmll = T.grad(static_marginal_log_likelihood(Phi2, y2, theta2), theta2)
grad_marginal_log_likelihood_theano = theano.function([Phi2, y2, theta2], gmll)







Phi3 = T.dmatrix('Phi3')
y3 = T.dvector('y3')
alpha = T.dscalar('alpha')
beta = T.dscalar('beta')

compute_blr_matrices_theano = theano.function([Phi3, y3, alpha, beta], static_compute_blr_matrices(Phi3,y3,alpha,beta))









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
		self.models = None



	def marginal_log_likelihood(self, theta):
		return(marginal_log_likelihood_theano(self.X_transformed, self.y, theta))

	def grad_marginal_log_likelihood(self, theta):
		return(grad_marginal_log_likelihood_theano(self.X_transformed, self.y, theta))

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
			self.X_transformed = self.basis_func(X)
		else:
			self.X_transformed = self.X

		self.y = y

		if do_optimize:
			if self.do_mcmc:

				if not self.burned:
					self.log_hypers = 2+2*np.random.rand(self.n_hypers,2).astype(theano.config.floatX)
					self.log_hypers = theano.shared(self.log_hypers)


					def mll(t):
						return(-5*batched_marginal_log_likelihood(self.X_transformed, self.y, t))

					self.sampler = HMC_sampler.new_from_shared_positions( self.log_hypers, mll, n_steps = self.chain_length)

					# Do a burn-in in the first iteration
					[self.sampler.draw() for i in range(int(np.ceil(self.burnin_steps/self.chain_length)))]
	
					self.burned = True
	
				# Start sampling
				self.hypers = np.exp(self.sampler.draw())
	
			else:
				# Optimize hyperparameters of the Bayesian linear regression
				
				res = optimize.fmin_bfgs(
					lambda t: -self.marginal_log_likelihood(t),
					5+np.random.rand(2),
					lambda t: -self.grad_marginal_log_likelihood(t),
				)
				self.hypers = [[np.exp(res[0]), np.exp(res[1])]]
				
		else:
			self.hypers = [[self.alpha, self.beta]]

		self.update_model()

	def update_model(self, hypers=None):
		self.hypers = self.hypers if hypers is None else hypers
		self.models = []
		for sample in self.hypers:
			alpha = sample[0]
			beta = sample[1]

			logger.debug("Alpha=%f ; Beta=%f" % (alpha, beta))
			self.models.append(compute_blr_matrices_theano(self.X_transformed, self.y, alpha,beta))

	def predict(self, X_test):
		r"""
		Returns the predictive mean and variance of the objective function at
		the given test points.

		Parameters
		----------
		X_test: np.ndarray (N, D)
			N input test points

		Returns
		----------
		np.array(N,)
			predictive mean
		np.array(N,)
			predictive variance

		"""
		if self.basis_func is not None:
			X_transformed = self.basis_func(X_test)
		else:
			X_transformed = X_test
			
		# Marginalise predictions over hyperparameters
		mu = np.zeros([len(self.hypers), X_transformed.shape[0]])
		var = np.zeros([len(self.hypers), X_transformed.shape[0]])
	
		for i, h in enumerate(self.hypers):
			mu[i] = np.dot(self.models[i][0].T, X_transformed.T)
			var[i] = 1. / h[1] + np.diag(np.dot(np.dot(X_transformed, self.models[i][1]), X_transformed.T))

		m = mu.mean(axis=0)
		v = var.mean(axis=0)
		# Clip negative variances and set them to the smallest
		# positive float value
		if v.shape[0] == 1:
			v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
		else:
			v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
			v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

		return m, v



N = 20
x = np.linspace(0,1,N)


Phi = quadratic_basis_func(x[:,None])

#Phi = linear_basis_func(x[:,None])

y = 0.5*x + 0 + np.random.randn(len(x))*0.02


model = BayesianLinearRegression(1,1000,do_mcmc=1,chain_length=2000)
model.train(Phi,y,True)












def plot_predictions( P = Phi):
	mu,var= model.predict(P)
	plt.scatter(x,y)
	plt.plot(x,mu)
	plt.fill_between(x, mu-np.sqrt(var), mu+np.sqrt(var),alpha=0.5)
	plt.show()


embed()
