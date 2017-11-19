import logging
import time
import numpy as np
import emcee
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer as HeNormal

from scipy import optimize

from robo.models.base_model import BaseModel
from robo.priors.bayesian_linear_regression_prior import BayesianLinearRegressionPrior
from robo.models.bayesian_linear_regression import BayesianLinearRegression
from robo.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_unnormalization


logger = logging.getLogger(__name__)



# def sharedX(X, dtype=theano.config.floatX, name=None):
#     return theano.shared(np.asarray(X, dtype=dtype), name=name)


# def smorms3(cost, params, lrate=1e-3, eps=1e-16, gather=False):
#     updates = []
#     optim_params = []
#     grads = T.grad(cost, params)
#
#     for p, grad in zip(params, grads):
#         mem = sharedX(p.get_value() * 0. + 1.)
#         g = sharedX(p.get_value() * 0.)
#         g2 = sharedX(p.get_value() * 0.)
#         if gather:
#             optim_params.append(mem)
#             optim_params.append(g)
#             optim_params.append(g2)
#
#         r_t = 1. / (mem + 1)
#         g_t = (1 - r_t) * g + r_t * grad
#         g2_t = (1 - r_t) * g2 + r_t * grad**2
#         p_t = p - grad * T.minimum(lrate, g_t * g_t / (g2_t + eps)) / \
#               (T.sqrt(g2_t + eps) + eps)
#         mem_t = 1 + mem * (1 - g_t * g_t / (g2_t + eps))
#
#         updates.append((g, g_t))
#         updates.append((g2, g2_t))
#         updates.append((p, p_t))
#         updates.append((mem, mem_t))
#     return updates


class DNGO(BaseModel):

    def __init__(self, batch_size=10, num_epochs=20000,
                 learning_rate=0.01, momentum=0.9,
                 adapt_epoch=5000, n_units_1=50, n_units_2=50, n_units_3=50,
                 alpha=1.0, beta=1000, prior=None, do_mcmc=True,
                 n_hypers=20, chain_length=2000, burnin_steps=2000,
                 normalize_input=True, normalize_output=True, rng=None):
        """
        Deep Networks for Global Optimization [1]. This module performs
        Bayesian Linear Regression with basis function extracted from a
        feed forward neural network.

        [1] J. Snoek, O. Rippel, K. Swersky, R. Kiros, N. Satish,
            N. Sundaram, M.~M.~A. Patwary, Prabhat, R.~P. Adams
            Scalable Bayesian Optimization Using Deep Neural Networks
            Proc. of ICML'15

        Parameters
        ----------
        batch_size: int
            Batch size for training the neural network
        num_epochs: int
            Number of epochs for training
        learning_rate: float
            Initial learning rate for SGD
        momentum: float
            Momentum for SGD
        adapt_epoch: int
            Defines after how many epochs the learning rate will be decayed by a factor 10
        n_units_1: int
            Number of units in layer 1
        n_units_2: int
            Number of units in layer 2
        n_units_3: int
            Number of units in layer 3
        alpha: float
            Hyperparameter of the Bayesian linear regression
        beta: float
            Hyperparameter of the Bayesian linear regression
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
        normalize_output : bool
            Zero mean unit variance normalization of the output values
        normalize_input : bool
            Zero mean unit variance normalization of the input values
        rng: np.random.RandomState
            Random number generator
        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        self.X = None
        self.y = None
        self.network = None
        self.alpha = alpha
        self.beta = beta
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        # MCMC hyperparameters
        self.do_mcmc = do_mcmc
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        if prior is None:
            self.prior = BayesianLinearRegressionPrior(rng=self.rng)
        else:
            self.prior = prior

        # Network hyper parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_learning_rate = learning_rate

        self.momentum = momentum
        self.n_units_1 = n_units_1
        self.n_units_2 = n_units_2
        self.n_units_3 = n_units_3
        self.adapt_epoch = adapt_epoch

        self.target_var = tf.placeholder(tf.float64)
        self.input_var = tf.placeholder(tf.float64)
        self.models = []
        self.sess = tf.Session()

    @BaseModel._check_shapes_train
    def train(self, X, y, do_optimize=True):
        """
        Trains the model on the provided data.

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
        start_time = time.time()

        # Normalize inputs
        if self.normalize_input:
            self.X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X)
        else:
            self.X = X

        # Normalize ouputs
        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)
        else:
            self.y = y

        self.y = self.y[:, None]

        # Check if we have enough points to create a minibatch otherwise use all data points
        if self.X.shape[0] <= self.batch_size:
            batch_size = self.X.shape[0]
        else:
            batch_size = self.batch_size

        # Create the neural network
        n_datapoints, n_inputs = X.shape
        self.input_var = tf.placeholder(tf.float64, shape = (None, n_inputs))

        self.network = self._build_net(self.input_var)
        # Get Prediction
        prediction = self.network[0]

        # Define loss function for training
        loss = tf.reduce_mean(tf.squared_difference(prediction,self.target_var))

        #self.learning_rate = self.init_learning_rate
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.init_learning_rate, # init learning rate
                                                        global_step, # global step
                                                        self.adapt_epoch, # decay step
                                                        0.1, # decay rate
                                                        staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)

        # Start training
        lc = np.zeros([self.num_epochs])
        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()

            train_err = 0
            train_batches = 0

            for batch in self.iterate_minibatches(self.X, self.y,
                                                  batch_size, shuffle=True):
                inputs, targets = batch
                self.sess.run(train_op, feed_dict={self.input_var: inputs, self.target_var: targets})
                # train_err += self.train_fn(inputs, targets)
                train_batches += 1

            lc[epoch] = train_err / train_batches
            logging.debug("Epoch {} of {}".format(epoch + 1, self.num_epochs))
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time
            logging.debug("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
            #logging.debug("Training loss:\t\t{:.5g}".format(train_err / train_batches))


        # Design matrix
        self.Theta = self.sess.run(self.network[1], feed_dict={self.input_var: self.X})

        if do_optimize:
            if self.do_mcmc:
                self.sampler = emcee.EnsembleSampler(self.n_hypers, 2,
                                                     self.marginal_log_likelihood)

                # Do a burn-in in the first iteration
                if not self.burned:
                    # Initialize the walkers by sampling from the prior
                    self.p0 = self.prior.sample_from_prior(self.n_hypers)
                    # Run MCMC sampling
                    self.p0, _, _ = self.sampler.run_mcmc(self.p0,
                                                          self.burnin_steps,
                                                          rstate0=self.rng)

                    self.burned = True

                # Start sampling
                pos, _, _ = self.sampler.run_mcmc(self.p0,
                                                  self.chain_length,
                                                  rstate0=self.rng)

                # Save the current position, it will be the startpoint in
                # the next iteration
                self.p0 = pos

                # Take the last samples from each walker
                self.hypers = np.exp(self.sampler.chain[:, -1])
            else:
                # Optimize hyperparameters of the Bayesian linear regression
                res = optimize.fmin(self.nll, np.random.rand(2))
                self.hypers = [[np.exp(res[0]), np.exp(res[1])]]
        else:

            self.hypers = [[self.alpha, self.beta]]

        logging.info("Hypers: %s" % self.hypers)
        self.models = []
        for sample in self.hypers:

            # Instantiate a model for each hyperparameter configuration
            model = BayesianLinearRegression(alpha=sample[0],
                                             beta=sample[1],
                                             basis_func=None)
            model.train(self.Theta, self.y[:, 0], do_optimize=False)

            self.models.append(model)

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
        if np.any((-5 > theta) + (theta > 10)):
            return -1e25

        alpha = np.exp(theta[0])
        beta = np.exp(theta[1])

        D = self.Theta.shape[1]
        N = self.Theta.shape[0]

        K = beta * np.dot(self.Theta.T, self.Theta)
        K += np.eye(self.Theta.shape[1]) * alpha
        K_inv = np.linalg.inv(K)
        m = beta * np.dot(K_inv, self.Theta.T)
        m = np.dot(m, self.y)

        mll = D / 2 * np.log(alpha)
        mll += N / 2 * np.log(beta)
        mll -= N / 2 * np.log(2 * np.pi)
        mll -= beta / 2. * np.linalg.norm(self.y - np.dot(self.Theta, m), 2)
        mll -= alpha / 2. * np.dot(m.T, m)
        mll -= 0.5 * np.log(np.linalg.det(K))

        l = mll + self.prior.lnprob(theta)

        return l

    def negative_mll(self, theta):
        """
        Returns the negative marginal log likelihood (for optimizing it with scipy).

        Parameters
        ----------
        theta: np.array(2,)
            The hyperparameter alpha and beta on a log scale

        Returns
        -------
        float
            negative lnlikelihood + prior
        """
        nll = -self.marginal_log_likelihood(theta)
        return nll

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[0] == targets.shape[0],\
               "The number of training points is not the same"
        if shuffle:
            indices = np.arange(inputs.shape[0])
            self.rng.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    @BaseModel._check_shapes_predict
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
        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        # Get features from the net

        theta = self.sess.run(self.network[1], feed_dict={self.input_var: X_})

        # Marginalise predictions over hyperparameters of the BLR
        mu = np.zeros([len(self.models), X_test.shape[0]])
        var = np.zeros([len(self.models), X_test.shape[0]])

        for i, m in enumerate(self.models):
            mu[i], var[i] = m.predict(theta)

        # See the algorithm runtime prediction paper by Hutter et al
        # for the derivation of the total variance
        m = np.mean(mu, axis=0)
        v = np.mean(mu ** 2 + var, axis=0) - m ** 2

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        if self.normalize_output:
                m = zero_mean_unit_var_unnormalization(m, self.y_mean, self.y_std)
                v *= self.y_std ** 2

        return m, v

    def _build_net(self, input_var, seed=None, dtype = tf.float64):

        #First Dense Layer
        network = tf.layers.dense(
            input_var,
            units = 50,
            kernel_initializer=HeNormal(factor=1.0, dtype=dtype, seed=seed),
            bias_initializer=tf.zeros_initializer(dtype=dtype),
            activation=tf.tanh
        )

        #Second Dense Layer
        network = tf.layers.dense(
            network,
            units = 50,
            kernel_initializer=HeNormal(factor=1.0, dtype=dtype, seed=seed),
            bias_initializer=tf.zeros_initializer(dtype=dtype),
            activation=tf.tanh
        )

        #Third Dense Layer
        network = tf.layers.dense(
            network,
            units = 50,
            kernel_initializer=HeNormal(factor=1.0, dtype=dtype, seed=seed),
            bias_initializer=tf.zeros_initializer(dtype=dtype),
            activation=tf.tanh
        )

        thirdLayer = network

        #Output Layer
        network = tf.layers.dense(
            network,
            units = 1,
            kernel_initializer=HeNormal(factor=1.0, dtype=dtype, seed=seed),
            bias_initializer=tf.zeros_initializer(dtype=dtype),
            activation= None # Linear
        )
        return network, thirdLayer

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        Returns
        ----------
        incumbent: ndarray (D,)
            current incumbent
        incumbent_value: ndarray (N,)
            the observed value of the incumbent
        """

        inc, inc_value = super(DNGO, self).get_incumbent()
        if self.normalize_input:
            inc = zero_mean_unit_var_unnormalization(inc, self.X_mean, self.X_std)

        if self.normalize_output:
            inc_value = zero_mean_unit_var_unnormalization(inc_value, self.y_mean, self.y_std)

        return inc, inc_value
