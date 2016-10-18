import time
import lasagne
import logging
import theano
import theano.tensor as T
import numpy as np


class SGDNet(object):

    def __init__(self, n_inputs,  n_epochs=100, l_rate=1e-3,
                 n_units_1=10, n_units_2=10, n_units_3=10,
                 noise_std=0.1, wd=1e-5, bsize=10, burn_in=1000,
                 precondition=True, normalize_output=True,
                 normalize_input=True, rng=None):
        """
        Constructor

        Parameters
        ----------
        n_inputs : int
            Number of input features
        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(100000))
        else:
            self.rng = rng
        lasagne.random.set_rng(self.rng)

        self.n_epochs = n_epochs
        self.l_rate = l_rate
        self.n_units_1 = n_units_1
        self.n_units_2 = n_units_2
        self.n_units_3 = n_units_3
        self.noise_std = noise_std
        self.wd = wd
        self.bsize = bsize
        self.burn_in = burn_in
        self.precondition = precondition
        self.is_trained = False
        self.normalize_output = normalize_output
        self.normalize_input = normalize_input

        self.Xt = T.matrix()
        self.Yt = T.matrix()

        self.X = None
        self.x_mean = None
        self.x_std = None
        self.Y = None
        self.y_mean = None
        self.y_std = None
        self.input_var = T.matrix('inputs')
        target_var = T.matrix('targets')

        self.network = self.get_net(n_inputs, self.input_var, self.n_units_1, self.n_units_2, self.n_units_3)

        prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = loss.mean()

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=l_rate)

        self.train_fn = theano.function([self.input_var, target_var], loss, updates=updates)

        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
        test_loss = test_loss.mean()

        self.val_fn = theano.function([self.input_var, target_var], test_loss)

    @staticmethod
    def get_net(n_inputs, input_var, n_units_1, n_units_2, n_units_3):
        network = lasagne.layers.InputLayer(shape=(None, n_inputs), input_var=input_var)

        network = lasagne.layers.DenseLayer(
            network,
            num_units=n_units_1,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(val=0.0),
            nonlinearity=lasagne.nonlinearities.tanh)
        network = lasagne.layers.dropout(network, p=.5)
        network = lasagne.layers.DenseLayer(
            network,
            num_units=n_units_2,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(val=0.0),
            nonlinearity=lasagne.nonlinearities.tanh)
        network = lasagne.layers.dropout(network, p=.5)
        network = lasagne.layers.DenseLayer(
            network,
            num_units=n_units_3,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(val=0.0),
            nonlinearity=lasagne.nonlinearities.tanh)
        network = lasagne.layers.dropout(network, p=.5)
        network = lasagne.layers.DenseLayer(
            network,
            num_units=1,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(val=0.0),
            nonlinearity=lasagne.nonlinearities.linear)

        return network

    def train(self, X, Y, X_valid, Y_valid):
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

        if self.normalize_input:
            self.X, self.x_mean, self.x_std = self.normalize_inputs(X)
        else:
            self.X = X

        if self.normalize_output:
            self.Y, self.y_mean, self.y_std = self.normalize_targets(Y)
        else:
            self.Y = Y

        # Discard old weights
        if self.is_trained:
            self.network = self.get_net(X.shape[1], self.input_var, self.n_units_1, self.n_units_2, self.n_units_3)
        logging.info("Starting training...")

        if X.shape[0] < self.bsize:
            batch_size = X.shape[0]
        else:
            batch_size = self.bsize

        for epoch in range(self.n_epochs):

            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(self.X, self.Y, batch_size, shuffle=True):
                inputs, targets = batch
                train_err += self.train_fn(inputs, targets)
                train_batches += 1

            logging.info("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.n_epochs, time.time() - start_time))
            logging.info("  training loss:\t\t{:.6f}".format(train_err / train_batches))

        val_err = 0

        val_batches = 0
        for batch in self.iterate_minibatches(X_valid, Y_valid, batch_size, shuffle=False):
            inputs, targets = batch
            err = self.val_fn(inputs, targets)
            val_err += err
            val_batches += 1
        logging.info("  valid loss:\t\t{:.6f}".format(val_err / val_batches))

        self.is_trained = True

        return val_err / val_batches

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

        if not self.is_trained:
            logging.error("Model is not trained!")
            return

        # Normalize input
        if self.normalize_input:
            X_, _, _ = self.normalize_inputs(X_test, self.x_mean, self.x_std)
        else:
            X_ = X_test

        l = lasagne.layers.get_all_layers(self.network)
        m = lasagne.layers.get_output(l, X_)[-1].eval()

        # denormalize output
        if self.normalize_output:
            m = self.denormalize(m, self.y_mean, self.y_std)

        return m[:, None]

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]


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
