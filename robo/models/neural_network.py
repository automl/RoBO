import time
import logging
import numpy as np

try:
    import lasagne
    import theano
    import theano.tensor as T
except ImportError as e:
    print(str(e))
    print("If you want to use Bayesian Neural Networks you have to install the following dependencies:")
    print("Theano (pip install theano)")
    print("Lasagne (pip install lasagne)")


class SGDNet(object):

    def __init__(self, n_inputs,  n_epochs=100,
                 error_threshold=0,
                 learning_rate=1e-3,
                 n_units=[10, 10, 10],
                 dropout=[0.5, 0.5, 0.5],
                 batch_size=10, shuffle_batches=False,
                 normalize_output=True,
                 normalize_input=True, rng=None):
        """
        Constructor

        Parameters
        ----------
        n_inputs : int
            Number of input features
        n_epochs : int
            maxmimum number of epochs
        learning_rate : float
            learing rate used for ADAM
        n_units : list of ints
            number of units in each layer, thus controls also the number of layers
        dropout : float or list of floats
            If it's a list, every element defines the dropout for each layer. If it's float
            the same dropout is applied at each layer
        batch_size : int
            size of the minibatches used during training
        shuffle_batches : boolean
            whether or not to permute the data points during training
        normalize_output : boolean
            whether or not the output should be scaled to zero mean, unit variance for training
        normalize_input : boolean
            whether or not the input should be scaled to zero mean, unit variance for training
        rng : numpy.random.RandomState
            the random number generator
        """

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng
        lasagne.random.set_rng(self.rng)

        self.n_inputs = int(n_inputs)
        self.n_epochs = int(n_epochs)
        self.error_threshold = error_threshold
        self.l_rate = learning_rate
        self.n_units = n_units

        # bring the dropout parameters in a common form
        try:
            if len(dropout) != len(n_units):
                raise ValueError("Number of dropout rates must match the number of layers")
            self.dropout = dropout
        except TypeError as e:
            self.dropout = [dropout] * len(n_units)
        except:
            raise

        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.is_trained = False
        self.normalize_output = normalize_output
        self.normalize_input = normalize_input

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.input_var = T.matrix('inputs')

        self.network = None

    def initialize_net(self):
        """
            creates the network and the associated loss and update functions
        """
        self.network = lasagne.layers.InputLayer(shape=(None, self.n_inputs), input_var=self.input_var)

        for n, p in zip(self.n_units, self.dropout):
            self.network = lasagne.layers.DenseLayer(
                self.network,
                num_units=n,
                W=lasagne.init.HeNormal(),
                b=lasagne.init.Constant(val=0.0),
                nonlinearity=lasagne.nonlinearities.tanh)
            self.network = lasagne.layers.dropout(self.network, p=p)

        self.network = lasagne.layers.DenseLayer(
            self.network,
            num_units=1,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(val=0.0),
            nonlinearity=lasagne.nonlinearities.linear)

        target_var = T.matrix('targets')

        prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = loss.mean()

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=self.l_rate)

        self.train_fn = theano.function([self.input_var, target_var], loss, updates=updates)

        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
        test_loss = test_loss.mean()

        self.val_fn = theano.function([self.input_var, target_var], test_loss)

        # g = T.grad(test_prediction, self.input_var)

        # self.gradient = theano.function([self.input_var], g)

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

        self.initialize_net()

        if self.normalize_input:
            X, self.x_mean, self.x_std = self.normalize_inputs(X)
        else:
            self.x_mean, self.x_std = None, None

        if self.normalize_output:
            Y, self.y_mean, self.y_std = self.normalize_targets(Y)
            
        else:
            self.Y, self.y_mean, self.y_std = Y, None, None

        logging.info("Starting training...")

        if X.shape[0] < self.batch_size:
            batch_size = X.shape[0]
        else:
            batch_size = self.batch_size

        for epoch in range(self.n_epochs):

            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(X, Y, batch_size, shuffle=self.shuffle_batches):
                inputs, targets = batch
                train_err += self.train_fn(inputs, targets)
                train_batches += 1

            logging.info("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.n_epochs, time.time() - start_time))
            logging.info("  training loss:\t\t{:.6f}".format(train_err / train_batches))

            if train_err/train_batches < self.error_threshold:
                break

        l = lasagne.layers.get_all_layers(self.network)
        m = lasagne.layers.get_output(l, X)[-1].eval()

    def validation_error(self, X_valid, Y_valid):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X_valid: np.ndarray (N, D)
            Validation data points. The dimensionality is (N, D),
            where N is the number of points and D is the number of features.
        Y_valid: np.ndarray (N, T)
            The corresponding target values.
            The dimensionality of Y is (N, T), where N has to
            match the number of points of X and T is the number of objectives
        """

        if X_valid.shape[0] < self.batch_size:
            batch_size = X_valid.shape[0]
        else:
            batch_size = self.batch_size

        if self.normalize_input:
            X_valid = self.normalize_inputs(X_valid, self.x_mean, self.x_std)[0]
        if self.normalize_output:
            Y_valid = self.normalize_targets(Y_valid, self.y_mean, self.y_std)[0]

        val_err = 0
        val_batches = 0

        for batch in self.iterate_minibatches(X_valid, Y_valid, batch_size, shuffle=False):
            inputs, targets = batch
            err = self.val_fn(inputs, targets)
            val_err += err
            val_batches += 1
        logging.info("  valid loss:\t\t{:.6f}".format(val_err / val_batches))
        return val_err / val_batches

    def predict(self, X_test):
        """
        makes predictions for all given points

        Parameters
        ----------
        X_test : numpy.array
            2d matrix with points as rows

        Returns
        ----------
        numpy.array(dtype=float)
            predictions for all points
        """

        if self.network is None:
            logging.error("Model is not trained!")
            return

        if self.normalize_input:
            X_test = self.normalize_inputs(X_test, self.x_mean, self.x_std)[0]

        l = lasagne.layers.get_all_layers(self.network)
        m = lasagne.layers.get_output(l, X_test)[-1].eval()

        # denormalize output
        if self.normalize_output:
            m = self.denormalize(m, self.y_mean, self.y_std)

        return m[:, None]

    # def predict_gradient(self, X):
    #
    #     fx = 1.
    #     fy = 1.
    #
    #     if self.normalize_input:
    #         X_test = self.normalize_inputs(X, self.x_mean, self.x_std)[0]
    #         fx = self.x_std
    #
    #     g = self.gradient(X)
    #
    #     if self.normalize_output:
    #         g = self.normalize_outputs(g, self.y_mean, self.y_std)[0]
    #         fy = self.y_std
    #
    #     return g * (fy/fx)

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        """
            helper function to quickly iterate over the data
        """
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
        x : numpy.array
            input vector
        mean: float
            mean used for normalization
        std: float
            standard deviation used for normalization

        Returns
        ----------
        float
            scaled x as input to the network
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
        y : numpy.array
            target values
        mean: float
            mean used for normalization
        std: float
            standard deviation used for normalization

        Returns
        ----------
        float
            scaled y as target output for the network
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
        y : numpy.array
            target values
        mean: float
            mean used for normalization
        std: float
            standard deviation used for normalization

        Returns
        ----------
        float
            scaled y as target output for the network

        """
        return mean + x * std
