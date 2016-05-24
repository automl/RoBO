import logging
import time
import numpy as np
import lasagne
import theano
import theano.tensor as T
import emcee
import sys
sys.path.insert(0, "/home/kleinaa/.local/lib/python2.7/site-packages/")

from scipy import optimize

from robo.models.base_model import BaseModel
from robo.priors.dngo_priors import DNGOPrior
from robo.models.bayesian_linear_regression import BayesianLinearRegression
from functools import partial


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def smorms3(cost, params, lrate=1e-3, eps=1e-16, gather=False):
    updates = []
    optim_params = []
    grads = T.grad(cost, params)

    for p, grad in zip(params, grads):
        mem = sharedX(p.get_value() * 0. + 1.)
        g = sharedX(p.get_value() * 0.)
        g2 = sharedX(p.get_value() * 0.)
        if gather:
            optim_params.append(mem)
            optim_params.append(g)
            optim_params.append(g2)

        r_t = 1. / (mem + 1)
        g_t = (1 - r_t) * g + r_t * grad
        g2_t = (1 - r_t) * g2 + r_t * grad**2
        p_t = p - grad * T.minimum(lrate, g_t * g_t / (g2_t + eps)) / \
              (T.sqrt(g2_t + eps) + eps)
        mem_t = 1 + mem * (1 - g_t * g_t / (g2_t + eps))

        updates.append((g, g_t))
        updates.append((g2, g2_t))
        updates.append((p, p_t))
        updates.append((mem, mem_t))
    
    return updates

class DNGO(BaseModel):

    def __init__(self, batch_size, num_epochs, learning_rate,
                 momentum, l2, adapt_epoch,
                 n_units_1=50, n_units_2=50, n_units_3=50,
                 alpha=1.0, beta=1000, do_optimize=True, do_mcmc=True,
                 prior=None, n_hypers=20, chain_length=2000,
                 burnin_steps=2000, *args, **kwargs):

        self.X = None
        self.Y = None
        self.network = None
        self.alpha = alpha
        self.beta = beta
        self.do_optimize = do_optimize
        
        # MCMC hyperparameters
        self.do_mcmc = do_mcmc
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps 
        if prior is None:
            self.prior = DNGOPrior()
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
        self.l2 = l2
        self.adapt_epoch = adapt_epoch

        self.target_var = T.matrix('targets')
        self.input_var = T.matrix('inputs')
        self.models = []

    def train(self, X, Y, **kwargs):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input datapoints. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        Y: np.ndarray (N, T)
            The corresponding target values.
            The dimensionality of Y is (N, T), where N has to
            match the number of points of X and T is the number of objectives
        """
        # Normalize inputs
        self.X = X
        self.X_mean = np.mean(X)
        self.X_std = np.std(X)
        self.norm_X = (X - self.X_mean) / self.X_std

        if self.X.shape[0] <= self.batch_size:
            batch_size = self.X.shape[0]
        else:
            batch_size = self.batch_size

        # Normalize ouputs
        self.Y_mean = np.mean(Y)
        self.Y_std = np.std(Y)
        self.Y = (Y - self.Y_mean) / self.Y_std
        #self.Y = Y
        start_time = time.time()

        # Create the neural network
        features = X.shape[1]

        self.learning_rate = theano.shared(np.array(self.init_learning_rate,
                                                dtype=theano.config.floatX))
        self.network = self._build_net(self.input_var, features)



        prediction = lasagne.layers.get_output(self.network)

        # Define loss function for training
        loss = T.mean(T.square(prediction - self.target_var)) / 0.001

        # Add l2 regularization for the weights
        l2_penalty = self.l2 * lasagne.regularization.regularize_network_params(
            self.network, lasagne.regularization.l2)
        loss += l2_penalty
        loss = loss.mean()

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        
        updates = lasagne.updates.adam(loss, params,
                                        learning_rate=self.learning_rate)


        logging.debug("... compiling theano functions")
        self.train_fn = theano.function([self.input_var, self.target_var], loss,
                                        updates=updates,
                                        allow_input_downcast=True)

        # Start training
        lc = np.zeros([self.num_epochs])
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()

            # Full pass over the training data:
            train_err = 0
            train_batches = 0

            for batch in self.iterate_minibatches(self.norm_X, self.Y,
                                            batch_size, shuffle=True):
                inputs, targets = batch
                train_err += self.train_fn(inputs, targets)
                train_batches += 1

            lc[epoch] = train_err / train_batches
            logging.debug("Epoch {} of {}".format(epoch + 1, self.num_epochs))
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time
            logging.debug("Epoch time {:.3f}s, "
                 "total time {:.3f}s".format(epoch_time, total_time))
            logging.debug("Training loss:\t\t{:.5g}".format(train_err / train_batches))

            #Adapt the learning rate
            if epoch % self.adapt_epoch == 0:
                print "Adapt lr %f " % (self.init_learning_rate * 0.1)
                self.learning_rate.set_value(
                            np.float32(self.init_learning_rate * 0.1))

        logging.debug("Learning curve")
        print(lc)

        # Design matrix
        layers = lasagne.layers.get_all_layers(self.network)
        self.Theta = lasagne.layers.get_output(layers[:-1], self.norm_X)[-1].eval()
        
        if self.do_optimize:
            if self.do_mcmc:
                self.sampler = emcee.EnsembleSampler(self.n_hypers,
                                                 2,
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
            model.train(self.Theta, self.Y, do_optimize=False)

            self.models.append(model)

    def marginal_log_likelihood(self, theta):
        
        if np.any((-5 > theta) + (theta > 10)):
            return -1e25

        alpha = np.exp(theta[0])
        beta = np.exp(theta[1])

        D = self.norm_X.shape[1]
        N = self.norm_X.shape[0]

        K = beta * np.dot(self.Theta.T, self.Theta)
        K += np.eye(self.Theta.shape[1]) * alpha**2
        K_inv = np.linalg.inv(K)
        m = beta * np.dot(K_inv, self.Theta.T)
        m = np.dot(m, self.Y)

        mll = D / 2 * np.log(alpha)
        mll += N / 2 * np.log(beta)
        mll -= N / 2 * np.log(2 * np.pi)
        mll -= beta / 2. * np.linalg.norm(self.Y - np.dot(self.Theta, m), 2)
        mll -= alpha / 2. * np.dot(m.T, m)
        mll -= 0.5 * np.log(np.linalg.det(K))
        param = np.array([theta[0], np.log(1 / np.exp(theta[1]))])
        l = mll + self.prior.lnprob(param)

        return l

    def nll(self, theta):
        nll = -self.marginal_log_likelihood(theta)
        return nll

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[0] == targets.shape[0],\
               "The number of training points is not the same"
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def update(self, X, y):
        X = np.append(self.X, X, axis=0)
        y = np.append(self.Y, y, axis=0)
        self.train(X, y)
        
    def predict(self, X_test, **kwargs):

        # Normalize input data to 0 mean and unit std
        X_ = (X_test - self.X_mean) /  self.X_std 

        # Get features from the net

        layers = lasagne.layers.get_all_layers(self.network)        
        theta = lasagne.layers.get_output(layers[:-1], X_)[-1].eval()

        # Marginalise predictions over hyperparameters of the BLR
        mu = np.zeros([len(self.models), X_test.shape[0]])
        var = np.zeros([len(self.models), X_test.shape[0]])
    
        for i, m in enumerate(self.models):
            mu[i], var[i] = m.predict(theta)

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

    def _build_net(self, input_var, features):

        network = lasagne.layers.InputLayer(shape=(None, features),
                                                input_var=input_var)

        # Define each layer
        network = lasagne.layers.DenseLayer(
#             lasagne.layers.dropout(network, p=0.1),
             network,
             num_units=self.n_units_1,
             W=lasagne.init.HeNormal(),
             b=lasagne.init.Constant(val=0.0),
             nonlinearity=lasagne.nonlinearities.tanh)

        network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=0.1),
            network,
            num_units=self.n_units_2,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(val=0.0),
            nonlinearity=lasagne.nonlinearities.tanh)

        network = lasagne.layers.DenseLayer(
 #            lasagne.layers.dropout(network, p=0.1),
             network,        
             num_units=self.n_units_3,
             W=lasagne.init.HeNormal(),
             b=lasagne.init.Constant(val=0.0),
             nonlinearity=lasagne.nonlinearities.tanh)

        # Define output layer
        network = lasagne.layers.DenseLayer(network,
                 num_units=1,
                 W=lasagne.init.HeNormal(),
                 b=lasagne.init.Constant(val=0.),
                 nonlinearity=lasagne.nonlinearities.linear)
        return network

    def predictive_gradients(self, X=None):
        """
        Calculates the predictive gradients (gradient of the prediction)

        Parameters
        ----------

        X: np.ndarray (N, D)
            The points to predict the gradient for

        Returns
        ----------
            The gradients at X
        """
        raise NotImplementedError()
        