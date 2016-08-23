import time
import lasagne
import logging
import theano
import theano.tensor as T
import numpy as np

from copy import deepcopy
from collections import deque
from sgmcmc.theano_mcmc import SGLDSampler
from sgmcmc.utils import floatX


class SGLDNet(object):

    def __init__(self, n_inputs, scale_grad, get_net=None,
                 n_nets=100, l_rate=1e-3, n_iters=5 * 10**4,
                 noise_std=0.1, wd=1e-5, bsize=10, burn_in=1000):
        """
        Constructor

        Parameters
        ----------
        n_inputs : int
            Number of input features
        """
        self.n_nets = n_nets
        self.l_rate = l_rate
        self.n_iters = n_iters
        self.noise_std = noise_std
        self.wd = wd
        self.bsize = bsize
        self.scale_grad = scale_grad
        self.burn_in = burn_in

        self.samples = deque(maxlen=n_nets)

        if get_net is None:
            self.net = self.get_net(n_inputs=n_inputs)
        else:
            self.net = get_net()

        self.sampler = SGLDSampler(precondition=True)

        Xt = T.matrix()
        Yt = T.matrix()

        nll, params = self.negativ_log_likelihood(self.net, Xt, Yt, Xsize=scale_grad, wd=wd, noise_std=noise_std)
        updates = self.sampler.prepare_updates(nll, params, self.l_rate, inputs=[Xt, Yt], scale_grad=scale_grad)
        err = T.sum(T.square(lasagne.layers.get_output(self.net, Xt) - Yt))
        self.compute_err = theano.function([Xt, Yt], err)
        self.single_predict = theano.function([Xt], lasagne.layers.get_output(self.net, Xt))
        self.train_fn = theano.function([Xt, Yt], nll, updates=updates)

        self.X = None
        self.x_mean = None
        self.x_std = None
        self.Y = None
        self.y_mean = None
        self.y_std = None

    @staticmethod
    def get_net(n_inputs):
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
        network = lasagne.layers.DenseLayer(
            fc_layer_3,
            num_units=1,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(val=0.0),
            nonlinearity=lasagne.nonlinearities.linear)

        return network

    def train(self, X, Y):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        Y: np.ndarray (N, T)
            The corresponding target values.
            The dimensionality of Y is (N, T), where N has to
            match the number of points of X and T is the number of objectives
        """

        self.X, self.x_mean, self.x_std = self.normalize_inputs(X)
        self.Y, self.y_mean, self.y_std = self.normalize_targets(Y)

        logging.info("Starting sampling")
        start_time = time.time()

        # Check if we have enough data points to form a minibatch
        # otherwise set the batchsize equal to the number of input points
        if self.X.shape[0] < self.bsize:
            self.bsize = self.X.shape[0]
            logging.info("Not enough datapoint to form a minibatch. "
                         "Set the batchsize to {}".format(self.bsize))

        i = 0
        while i < self.n_iters and len(self.samples) < self.n_nets:
            start = (i * self.bsize) % (self.X.shape[0])

            xmb = floatX(self.X[start:start + self.bsize])
            ymb = floatX(self.Y[start:start + self.bsize]).reshape((-1, 1))

            _, nll_value = self.sampler.step(xmb, ymb)

            if i % 1000 == 0:
                total_err = self.compute_err(floatX(self.X), floatX(self.Y).reshape(-1, 1))
                t = time.time() - start_time

                logging.info("Iter {} : NLL = {} MSE = {} "
                             "Collected samples= {} Time = {}".format(i,
                                                                      nll_value,
                                                                      total_err / self.X.shape[0],
                                                                      len(self.samples), t))
            if i % 200 == 0 and i >= self.burn_in:
                self.samples.append(lasagne.layers.get_all_param_values(self.net))

            i += 1

    def negativ_log_likelihood(self, net, X, Y, Xsize=1, wd=1, noise_std=0.1):
        """
        Negative log likelihood of the data

        Parameters
        ----------
        n_inputs : int
            Number of input features

        Returns
        ----------
        float
            lnlikelihood + prior
        """

        all_params = lasagne.layers.get_all_params(net, trainable=True)

        out = lasagne.layers.get_output(net, X)
        err = T.mean(T.square(Y - out) / (noise_std ** 2))

        prior_nll = 0.
        for p in all_params:
            prior_nll += T.sum(-T.square(p) * 0.5 * wd)
        nll = err - prior_nll / T.cast(Xsize, dtype=theano.config.floatX)
        return nll, all_params

    def predict(self, X_test):
        """
        Negative log likelihood of the data

        Parameters
        ----------
        n_inputs : int
            Number of input features

        Returns
        ----------
        float
            lnlikelihood + prior
        """
        # Normalize input
        X_, _, _ = self.normalize_inputs(X_test, self.x_mean, self.x_std)
        p = []
        for sample in self.samples:
            lasagne.layers.set_all_param_values(self.net, sample)
            out = self.single_predict(X_)[:, 0]
            p.append(out)

        m = np.mean(np.asarray(p), axis=0)
        v = np.var(np.asarray(p), axis=0)

        # denormalize output
        m = self.denormalize(m, self.y_mean, self.y_std)

        return m[:, None], v[:, None]

    @staticmethod
    def normalize_inputs(x, mean=None, std=None):
        """
        normalize_inputs

        Parameters
        ----------
        n_inputs : int
            Number of input features

        Returns
        ----------
        float
            lnlikelihood + prior
        """
        if mean is None:
            mean = np.mean(x, axis=0)
        if std is None:
            std = np.std(x, axis=0)
        return (x - mean) / std, mean, std

    @staticmethod
    def normalize_targets(y, mean=None, std=None):
        """
        normalize_targets

        Parameters
        ----------
        n_inputs : int
            Number of input features

        Returns
        ----------
        float
            lnlikelihood + prior
        """
        if mean is None:
            mean = np.mean(y, axis=0)
        if std is None:
            std = np.std(y, axis=0)
        return (y - mean) / std, mean, std

    @staticmethod
    def denormalize(x, mean, std):
        """
        denormalize

        Parameters
        ----------
        x : np.ndarray
            Data point

x : np.ndarray
            Data point

        Returns
        ----------
        float
            lnlikelihood + prior
        """
        return mean + x * std


class LCNet(SGLDNet):
    @staticmethod
    def normalize_inputs(x, mean=None, std=None):
        if mean is None:
            mean = np.mean(x, axis=0)
        if std is None:
            std = np.std(x, axis=0)

        x_norm = deepcopy(x)
        x_norm[:, :-1] = (x[:, :-1] - mean[:-1]) / std[:-1]
        return x_norm, mean, std

