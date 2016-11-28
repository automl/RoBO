import time
import logging
import numpy as np

from collections import deque

from robo.models.base_model import BaseModel
from robo.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_unnormalization

try:
    import theano
    import theano.tensor as T
    import lasagne
    from sgmcmc.theano_mcmc import SGLDSampler, SGHMCSampler
    from sgmcmc.utils import floatX
    from sgmcmc.bnn.priors import WeightPrior, LogVariancePrior
    from sgmcmc.bnn.lasagne_layers import AppendLayer

except ImportError as e:
    print(str(e))
    print("If you want to use Bayesian Neural Networks you have to install the following dependencies:")
    print("Theano (pip install theano)")
    print("Lasagne (pip install lasagne)")
    print("sgmcmc (see https://github.com/stokasto/sgmcmc)")


def get_default_net(n_inputs):
    l_in = lasagne.layers.InputLayer(shape=(None, n_inputs))

    fc_layer_1 = lasagne.layers.DenseLayer(
        l_in,
        num_units=50,
        W=lasagne.init.HeNormal(),
        b=lasagne.init.Constant(val=0.0),
        nonlinearity=lasagne.nonlinearities.tanh)
    fc_layer_2 = lasagne.layers.DenseLayer(
        fc_layer_1,
        num_units=50,
        W=lasagne.init.HeNormal(),
        b=lasagne.init.Constant(val=0.0),
        nonlinearity=lasagne.nonlinearities.tanh)
    fc_layer_3 = lasagne.layers.DenseLayer(
        fc_layer_2,
        num_units=50,
        W=lasagne.init.HeNormal(),
        b=lasagne.init.Constant(val=0.0),
        nonlinearity=lasagne.nonlinearities.tanh)
    l_out = lasagne.layers.DenseLayer(
        fc_layer_3,
        num_units=1,
        W=lasagne.init.HeNormal(),
        b=lasagne.init.Constant(val=0.0),
        nonlinearity=lasagne.nonlinearities.linear)

    network = AppendLayer(l_out, num_units=1, b=lasagne.init.Constant(np.log(1e-3)))

    return network


class BayesianNeuralNetwork(BaseModel):

    def __init__(self, sampling_method="sghmc",
                 n_nets=100, l_rate=1e-3,
                 mdecay=5e-2, n_iters=5 * 10**4,
                 bsize=20, burn_in=1000,
                 precondition=True, normalize_output=True,
                 normalize_input=True, rng=None, get_net=get_default_net):
        """
        Bayesian Neural Networks use Bayesian methods to estimate the posterior distribution of a neural
        network's weights. This allows to also predict uncertainties for test points and thus makes
        Bayesian Neural Networks suitable for Bayesian optimization.

        This module uses stochastic gradient MCMC methods to sample from the posterior distribution together See [1]
        for more details.

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            Bayesian Optimization with Robust Bayesian Neural Networks.
            In Advances in Neural Information Processing Systems 29 (2016).

        Parameters
        ----------
        sampling_method : str
            Determines the MCMC strategy:
            "sghmc" = Stochastic Gradient Hamiltonian Monte Carlo
            "sgld" = Stochastic Gradient Langevin Dynamics

        n_nets : int
            The number of samples (weights) that are drawn from the posterior

        l_rate : float
            The step size parameter for SGHMC

        mdecay : float
            Decaying term for the momentum in SGHMC

        n_iters : int
            Number of MCMC sampling steps without burn in

        bsize : int
            Batch size to form a mini batch

        burn_in : int
            Number of burn-in steps before the actual MCMC sampling begins

        precondition : bool
            Turns on / off preconditioning. See [1] for more details

        normalize_input : bool
            Turns on / off zero mean unit variance normalization of the input data

        normalize_output : bool
            Turns on / off zero mean unit variance normalization of the output data

        rng : np.random.RandomState()
            Random number generator

        get_net : func
            function that returns a network specification.

        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(100000))
        else:
            self.rng = rng

        lasagne.random.set_rng(self.rng)

        self.sampling_method = sampling_method
        self.n_nets = n_nets
        self.l_rate = l_rate
        self.mdecay = mdecay
        self.n_iters = n_iters
        self.bsize = bsize
        self.burn_in = burn_in
        self.precondition = precondition
        self.is_trained = False
        self.normalize_output = normalize_output
        self.normalize_input = normalize_input
        self.get_net = get_net

        self.samples = deque(maxlen=n_nets)

        self.variance_prior = LogVariancePrior(1e-6, prior_out_std_prec=0.01)
        self.weight_prior = WeightPrior(alpha=1., beta=1.)

        self.Xt = T.matrix()
        self.Yt = T.matrix()

        self.X = None
        self.x_mean = None
        self.x_std = None
        self.y = None
        self.y_mean = None
        self.y_std = None

    @BaseModel._check_shapes_train
    def train(self, X, y, *args, **kwargs):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.

        """

        # Clear old samples
        start_time = time.time()

        self.net = self.get_net(n_inputs=X.shape[1])

        nll, mse = self.negativ_log_likelihood(self.net, self.Xt, self.Yt, X.shape[0], self.weight_prior, self.variance_prior)
        params = lasagne.layers.get_all_params(self.net, trainable=True)

        seed = self.rng.randint(1, 100000)
        srng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed)

        if self.sampling_method == "sghmc":
            self.sampler = SGHMCSampler(rng=srng, precondition=self.precondition, ignore_burn_in=False)
        elif self.sampling_method == "sgld":
            self.sampler = SGLDSampler(rng=srng, precondition=self.precondition)
        else:
            logging.error("Sampling Strategy % does not exist!" % self.sampling_method)

        self.compute_err = theano.function([self.Xt, self.Yt], [mse, nll])
        self.single_predict = theano.function([self.Xt], lasagne.layers.get_output(self.net, self.Xt))

        self.samples.clear()

        if self.normalize_input:
            self.X, self.x_mean, self.x_std = zero_mean_unit_var_normalization(X)
        else:
            self.X = X

        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)
        else:
            self.y = y

        self.sampler.prepare_updates(nll, params, self.l_rate, mdecay=self.mdecay,
                                     inputs=[self.Xt, self.Yt], scale_grad=X.shape[0])

        logging.info("Starting sampling")

        # Check if we have enough data points to form a minibatch
        # otherwise set the batchsize equal to the number of input points
        if self.X.shape[0] < self.bsize:
            self.bsize = self.X.shape[0]
            logging.error("Not enough datapoint to form a minibatch. "
                          "Set the batchsize to {}".format(self.bsize))

        i = 0
        while i < self.n_iters and len(self.samples) < self.n_nets:
            if self.X.shape[0] == self.bsize:
                start = 0
            else:
                start = np.random.randint(0, self.X.shape[0] - self.bsize)

            xmb = floatX(self.X[start:start + self.bsize])
            ymb = floatX(self.y[start:start + self.bsize, None])

            if i < self.burn_in:
                _, nll_value = self.sampler.step_burn_in(xmb, ymb)
            else:
                _, nll_value = self.sampler.step(xmb, ymb)

            if i % 1000 == 0:
                total_err, total_nll = self.compute_err(floatX(self.X), floatX(self.y).reshape(-1, 1))
                t = time.time() - start_time

                logging.info("Iter {} : NLL = {} MSE = {} "
                             "Collected samples= {} Time = {}".format(i,
                                                                      total_nll,
                                                                      total_err,
                                                                      len(self.samples), t))
            if i % 100 == 0 and i >= self.burn_in:
                self.samples.append(lasagne.layers.get_all_param_values(self.net))

            i += 1
        self.is_trained = True

    def negativ_log_likelihood(self, f_net, X, y, n_examples, weight_prior, variance_prior):

        f_out = lasagne.layers.get_output(f_net, X)
        f_mean = f_out[:, 0].reshape((-1, 1))

        f_log_var = f_out[:, 1].reshape((-1, 1))

        f_var_inv = 1. / (T.exp(f_log_var) + 1e-16)
        mse = T.square(y - f_mean)
        log_like = T.sum(T.sum(-mse * (0.5 * f_var_inv) - 0.5 * f_log_var, axis=1))
        # scale by batch size to make this work nicely with the updaters above
        log_like /= T.cast(X.shape[0], theano.config.floatX)
        # scale the priors by the dataset size for the same reason
        # prior for the variance
        tn_examples = T.cast(n_examples, theano.config.floatX)
        log_like += variance_prior.log_like(f_log_var, n_examples) / tn_examples
        # prior for the weights
        params = lasagne.layers.get_all_params(f_net, trainable=True)
        log_like += weight_prior.log_like(params) / tn_examples

        return -log_like, T.mean(mse)

    @BaseModel._check_shapes_predict
    def predict(self, X_test, return_individual_predictions=False, *args, **kwargs):
        """
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points

        return_individual_predictions: bool
            If set to true than the individual predictions of all samples are returned.

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance

        """

        if not self.is_trained:
            logging.error("Model is not trained!")
            return

        # Normalize input
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.x_mean, self.x_std)
        else:
            X_ = X_test

        f_out = []
        theta_noise = []
        for sample in self.samples:
            lasagne.layers.set_all_param_values(self.net, sample)
            out = self.single_predict(X_)
            f_out.append(out[:, 0])
            theta_noise.append(np.exp(out[:, 1]))

        f_out = np.asarray(f_out)
        theta_noise = np.asarray(theta_noise)
        
        if return_individual_predictions:
            return f_out, theta_noise

        m = np.mean(f_out, axis=0)
        # Total variance
        # v = np.mean(f_out ** 2 + theta_noise, axis=0) - m ** 2
        v = np.mean((f_out - m) ** 2, axis=0)

        if self.normalize_output:
            m = zero_mean_unit_var_unnormalization(m, self.y_mean, self.y_std)
            v *= self.y_std ** 2

        return m, v

    def sample_functions(self, X_test, n_funcs=1):
        """
        Samples F function values from the current posterior at the N
        specified test point.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        n_funcs: int
            Number of function values that are drawn at each test point.

        Returns
        ----------
        np.array(F, N)
            The F function values drawn at the N test points.
        """
        if self.normalize_input:
            X_test_norm, _, _ = zero_mean_unit_var_normalization(X_test, self.x_mean, self.x_std)
        else:
            X_test_norm = X_test
        f = np.zeros([n_funcs, X_test_norm.shape[0]])
        for i in range(n_funcs):
            lasagne.layers.set_all_param_values(self.net, self.samples[i])
            out = self.single_predict(X_test_norm)[:, 0]
            if self.normalize_output:
                f[i, :] = zero_mean_unit_var_unnormalization(out, self.y_mean, self.y_std)
            else:
                f[i, :] = out

        return f

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
        inc, inc_value = super(BayesianNeuralNetwork, self).get_incumbent()
        if self.normalize_input:
            inc = zero_mean_unit_var_unnormalization(inc, self.x_mean, self.x_std)

        if self.normalize_output:
            inc_value = zero_mean_unit_var_unnormalization(inc_value, self.y_mean, self.y_std)

        return inc, inc_value
