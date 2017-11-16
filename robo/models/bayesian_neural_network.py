# vim: foldmethod=marker

#  Imports {{{ #
from collections import deque
import itertools
import logging
from time import time
import numpy as np
import tensorflow as tf

from pysgmcmc.models.base_model import (
    BaseModel,
    zero_mean_unit_var_normalization,
    zero_mean_unit_var_unnormalization
)

from pysgmcmc.sampling import Sampler
from pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule

from pysgmcmc.data_batches import generate_batches
from pysgmcmc.tensor_utils import safe_divide

#  }}}  Imports #


#  Default Network Architecture {{{ #

def get_default_net(inputs, seed=None, dtype=tf.float64):
    from tensorflow.contrib.layers import variance_scaling_initializer as HeNormal
    fc_layer_1 = tf.layers.dense(
        inputs, units=50, activation=tf.tanh,
        kernel_initializer=HeNormal(factor=1.0, dtype=dtype, seed=seed),
        bias_initializer=tf.zeros_initializer(dtype=dtype),
        name="fc_layer_1"
    )

    fc_layer_2 = tf.layers.dense(
        fc_layer_1, units=50, activation=tf.tanh,
        kernel_initializer=HeNormal(factor=1.0, dtype=dtype, seed=seed),
        bias_initializer=tf.zeros_initializer(dtype=dtype),
        name="fc_layer_2"
    )

    fc_layer_3 = tf.layers.dense(
        fc_layer_2, units=50, activation=tf.tanh,
        kernel_initializer=HeNormal(factor=1.0, dtype=dtype, seed=seed),
        bias_initializer=tf.zeros_initializer(dtype=dtype),
        name="fc_layer_3"
    )

    layer_4 = tf.layers.dense(
        fc_layer_3, units=1, activation=None,  # linear activation
        kernel_initializer=HeNormal(factor=1.0, dtype=dtype, seed=seed),
        bias_initializer=tf.zeros_initializer(dtype=dtype),
        name="fc_layer_4"
    )

    output_bias = tf.Variable(
        [[np.log(1e-3)]], dtype=dtype,
        name="output_bias"
    )

    output = tf.concat(
        [layer_4, tf.ones_like(layer_4, dtype=dtype) * output_bias],
        axis=1,
        name="Network_Output"
    )

    return output

#  }}} Default Network Architecture #


#  Priors {{{ #


def log_variance_prior_log_like(log_var, mean=1e-6, var=0.01, dtype=tf.float64):
    """
    Prior on the log predicted variance.

    Parameters
    ----------
    log_var : tensorflow.Tensor
        TODO:DOKU

    mean : float, optional
        Actual mean on a linear scale.
        Defaults to `1e-6`.

    var : float, optional
        Variance on a log scale.
        Defaults to `0.01`.

    dtype : tensorflow.DType, optional

    Returns
    -------
    log_like : tensorflow.Tensor
        TODO: DOKU

    """
    mean = tf.constant(mean, name="log_variance_prior_mean", dtype=dtype)
    var = tf.constant(var, name="log_variance_prior_var", dtype=dtype)

    return tf.reduce_mean(tf.reduce_sum(
        safe_divide(-tf.square(log_var - tf.log(mean)), (2. * var)) - 0.5 *
        tf.log(var), axis=1), name="variance_prior_log_like")


def weight_prior_log_like(parameters, wdecay=1., dtype=tf.float64):
    """
    Prior on the weights.

    Parameters
    ----------
    parameters : list of tensorflow.Variable objects

    weight_decay : float
        TODO DOKU
        Defaults to `1.`.

    dtype : tensorflow.DType, optional
        TODO DOKU
        Defaults to `tf.float64`

    Returns
    ----------
    log_like: tensorflow.Tensor

    """
    Wdecay = tf.constant(wdecay, name="wdecay", dtype=dtype)

    log_like = tf.convert_to_tensor(0., name="ll", dtype=dtype)
    n_params = tf.convert_to_tensor(0., name="n_params", dtype=dtype)

    for parameter in parameters:
        log_like += tf.reduce_sum(-Wdecay * 0.5 * tf.square(parameter))
        n_params += tf.cast(
            tf.reduce_prod(tf.to_float(parameter.shape)), dtype=dtype
        )
    return safe_divide(log_like, n_params, name="weight_prior_log_like")


#  }}} Priors #


class BayesianNeuralNetwork(object):
    def __init__(self, session, sampling_method=Sampler.SGHMC,
                 get_net=get_default_net,
                 batch_generator=generate_batches,
                 batch_size=20,
                 stepsize_schedule=ConstantStepsizeSchedule(np.sqrt(1e-4)),
                 n_nets=100, n_iters=50000,
                 burn_in_steps=1000, sample_steps=100,
                 normalize_input=True, normalize_output=True,
                 seed=None, dtype=tf.float64, **sampler_kwargs):
        """
        Bayesian Neural Networks use Bayesian methods to estimate the posterior
        distribution of a neural network's weights. This allows to also
        predict uncertainties for test points and thus makes Bayesian Neural
        Networks suitable for Bayesian optimization.

        This module uses stochastic gradient MCMC methods to sample
        from the posterior distribution.

        See [1] for more details.

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            Bayesian Optimization with Robust Bayesian Neural Networks.
            In Advances in Neural Information Processing Systems 29 (2016).

        Parameters
        ----------
        session: tensorflow.Session
            A `tensorflow.Session` object used to delegate computations
            performed in this network over to `tensorflow`.

        sampling_method : Sampler, optional
            Method used to sample networks for this BNN.
            Defaults to `Sampler.SGHMC`.

        n_nets: int, optional
            Number of nets to sample during training (and use to predict).
            Defaults to `100`.

        stepsize_schedule : pysgmcmc.stepsize_schedules.StepsizeSchedule
            Iterator class that produces a stream of stepsize values that
            we can use during sampling.
            See also: `pysgmcmc.stepsize_schedules`

        mdecay: float, optional
            Momentum decay per time-step (parameter for SGHMCSampler).
            Defaults to `0.05`.

        n_iters: int, optional
            Total number of iterations of the sampler to perform.
            Defaults to `50000`

        batch_size: int, optional
            Number of datapoints to include in each minibatch.
            Defaults to `20` datapoints per minibatch.

        burn_in_steps: int, optional
            Number of burn-in steps to perform
            Defaults to `1000`.

        sample_steps: int, optional
            Number of sample steps to perform.
            Defaults to `100`.

        normalize_input: bool, optional
            Specifies whether or not input data should be normalized.
            Defaults to `True`

        normalize_output: bool, optional
            Specifies whether or not outputs should be normalized.
            Defaults to `True`

        get_net: callable, optional
            Callable that returns a network specification.
            Expected inputs are a `tensorflow.Placeholder` object that
            serves as feedable input to the network and an integer random seed.
            Expected return value is the networks final output.
            Defaults to `get_default_net`.

        batch_generator: callable, optional
            TODO: DOKU
            NOTE: Generator callable with signature like generate_batches that
            yields feedable dicts of minibatches.

        seed: int, optional
            Random seed to use in this BNN.
            Defaults to `None`.

        dtype : tf.DType, optional
            Tensorflow datatype to use for internal representation.
            Defaults to `None`.

        """

        # Sanitize inputs
        assert isinstance(n_nets, int)
        assert isinstance(n_iters, int)
        assert isinstance(burn_in_steps, int)
        assert isinstance(sample_steps, int)
        assert isinstance(batch_size, int)

        assert isinstance(dtype, tf.DType)

        assert n_nets > 0
        assert n_iters > 0
        assert burn_in_steps >= 0
        assert sample_steps > 0
        assert batch_size > 0

        assert callable(get_net)
        assert callable(batch_generator)

        assert hasattr(stepsize_schedule, "update")
        assert hasattr(stepsize_schedule, "__next__")

        if not Sampler.is_supported(sampling_method):
            raise ValueError(
                "'BayesianNeuralNetwork.__init__' received unsupported input "
                "for parameter 'sampling_method'. Input was: {input}.\n"
                "Supported sampling methods are enumerated in "
                "'Sampler' enum type.".format(input=sampling_method)
            )

        self.sampling_method = sampling_method

        self.stepsize_schedule = stepsize_schedule

        self.get_net = get_net
        self.batch_generator = batch_generator

        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        self.n_nets = n_nets
        self.n_iters = n_iters

        self.batch_size = batch_size

        self.sampler_kwargs = sampler_kwargs

        self.burn_in_steps = burn_in_steps
        self.sample_steps = sample_steps

        self.samples = deque(maxlen=n_nets)

        self.seed = seed

        self.dtype = dtype

        self.session = session

        self.is_trained = False
        self.set_up_train = True

    def _train_set_up(self, X, y):

        self.X, self.y = X, y

        if self.normalize_input:
            self.X, self.x_mean, self.x_std = zero_mean_unit_var_normalization(self.X)

        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(self.y)

        n_datapoints, n_inputs = X.shape

        # set up placeholders for data minibatches
        self.X_Minibatch = tf.placeholder(shape=(None, n_inputs),
                                          dtype=self.dtype,
                                          name="X_Minibatch")
        self.Y_Minibatch = tf.placeholder(dtype=self.dtype, name="Y_Minibatch")

        # set up tensors for negative log likelihood and mean squared error
        self.Nll, self.Mse = self.negative_log_likelihood(
            X=self.X_Minibatch, Y=self.Y_Minibatch
        )

        self.network_params = tf.trainable_variables()

        self.sampler_kwargs.update({
            "params": self.network_params,
            "cost_fun": lambda *_: self.Nll,
            "batch_generator": self.batch_generator(
                x=self.X, x_placeholder=self.X_Minibatch,
                y=self.y, y_placeholder=self.Y_Minibatch,
                batch_size=self.batch_size,
                seed=self.seed
            ),
            "session": self.session,
            "seed": self.seed,
            "dtype": self.dtype,
            "stepsize_schedule": self.stepsize_schedule,
        })
        if Sampler.is_burn_in_mcmc(self.sampling_method):
            # Not always used, only for `pysgmcmc.sampling.BurnInMCMCSampler`
            # subclasses.
            self.sampler_kwargs.update({
                "scale_grad": n_datapoints,
                "burn_in_steps": self.burn_in_steps,
            })

        # NOTE: Burn_in_steps might not be a necessary parameter anymore,
        # if we find that some samplers do not need it.
        # In this case, we might get rid of it and make users specify it
        # as part of `sampler_args` instead.

        self.sampler = Sampler.get_sampler(
            self.sampling_method, **self.sampler_kwargs
        )

        self.set_up_train = False

    def negative_log_likelihood(self, X, Y):
        """ Compute the negative log likelihood of the
            current network parameters with respect to inputs `X` with
            labels `Y`.

        Parameters
        ----------
        X : tensorflow.Placeholder
            Placeholder for input datapoints.

        Y : tensorflow.Placeholder
            Placeholder for input labels.

        Returns
        -------
        neg_log_like: tensorflow.Tensor
            Negative log likelihood of the current network parameters with
            respect to inputs `X` with labels `Y`.


        mse: tensorflow.Tensor
            Mean squared error of the current network parameters
            with respect to inputs `X` with labels `Y`.

        """

        self.net_output = self.get_net(inputs=X, seed=self.seed, dtype=self.dtype)

        f_mean = tf.reshape(self.net_output[:, 0], shape=(-1, 1))
        f_log_var = tf.reshape(self.net_output[:, 1], shape=(-1, 1))

        f_var_inv = 1. / (tf.exp(f_log_var) + 1e-16)

        mse = tf.square(Y - f_mean)

        log_like = tf.reduce_sum(
            tf.reduce_sum(-mse * (0.5 * f_var_inv) - 0.5 * f_log_var, axis=1)
        )

        # scale by batch size to make this work nicely with the updaters above
        log_like = log_like / tf.constant(self.batch_size, dtype=self.dtype)

        # scale the priors by the dataset size for the same reason
        n_examples = tf.constant(self.X.shape[0], self.dtype, name="n_examples")

        # prior for the variance
        log_like += log_variance_prior_log_like(f_log_var, dtype=self.dtype) / n_examples

        # prior for the weights
        log_like += weight_prior_log_like(tf.trainable_variables(), dtype=self.dtype) / n_examples

        return -log_like, tf.reduce_mean(mse)

    @BaseModel._check_shapes_predict
    def train(self, X, y, *args, **kwargs):
        """ Train our Bayesian Neural Network using input datapoints `X`
            with corresponding labels `y`.

        Parameters
        ----------
        X : numpy.ndarray (N, D)
            Input training datapoints.

        y : numpy.ndarray (N,)
            Input training labels.
        """

        start_time = time()
        # remove any leftover samples from previous "train" calls
        self.samples.clear()

        if self.set_up_train:
            # only set up training graph once
            self._train_set_up(X, y)

        self.session.run(tf.global_variables_initializer())

        logging.info("Starting sampling")

        def log_full_training_error(iteration_index, is_sampling: bool):
            """ Compute the error of our last sampled network parameters
                on the full training dataset and use `logging.info` to
                log it. The boolean flag `sampling` is used to determine
                whether we are already collecting sampled networks, in which
                case additional info is logged using `logging.info`.

            Parameters
            ----------
            is_sampling : bool
                Boolean flag that specifies if we are already
                collecting samples or if we are still doing burn-in steps.
                If set to `True` we will also log the total number
                of samples collected thus far.

            """
            total_nll, total_mse = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: self.X,
                    self.Y_Minibatch: self.y.reshape(-1, 1)
                }
            )
            seconds_elapsed = time() - start_time
            if is_sampling:
                logging.info("Iter {:8d} : NLL = {:.4e} MSE = {:.4e} "
                             "Samples = {} Time = {:5.2f}".format(iteration_index,
                                                                  float(total_nll),
                                                                  float(total_mse),
                                                                  len(self.samples),
                                                                  seconds_elapsed))
            else:
                logging.info("Iter {:8d} : NLL = {:.4e} MSE = {:.4e} "
                             "Time = {:5.2f}".format(
                                 iteration_index, float(total_nll),
                                 float(total_mse), seconds_elapsed))

        logging_intervals = {"burn-in": 512, "sampling": self.sample_steps}

        sample_chain = itertools.islice(self.sampler, self.n_iters)

        for iteration_index, (parameter_values, _) in enumerate(sample_chain):

            burning_in = iteration_index <= self.burn_in_steps

            if burning_in and iteration_index % logging_intervals["burn-in"] == 0:
                log_full_training_error(
                    iteration_index=iteration_index, is_sampling=False
                )

            if not burning_in and iteration_index % logging_intervals["sampling"] == 0:
                log_full_training_error(
                    iteration_index=iteration_index, is_sampling=True
                )

                # collect sample
                self.samples.append(parameter_values)

                if len(self.samples) == self.n_nets:
                    # collected enough sample networks, stop iterating
                    break

        self.is_trained = True

    def compute_network_output(self, params, input_data):
        """ Compute and return the output of the network when
            parameterized with `params` on `input_data`.

        Parameters
        ----------
        params : list of ndarray objects
            List of parameter values (ndarray)
            for each tensorflow.Variable parameter of our network.

        input_data : ndarray (N, D)
            Input points to compute the network output for.

        Returns
        -------
        network_output: ndarray (N,)
            Output of the network parameterized with `params`
            on the given `input_data` points.
        """

        feed_dict = dict(zip(self.network_params, params))
        feed_dict[self.X_Minibatch] = input_data
        return self.session.run(self.net_output, feed_dict=feed_dict)

    @BaseModel._check_shapes_predict
    def predict(self, X_test, return_individual_predictions=False, *args, **kwargs):
        """
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test datapoints.

        return_individual_predictions: bool
            If set to `True` than the individual predictions of
            all samples are returned.

        Returns
        ----------
        mean: np.array(N,)
            predictive mean

        variance: np.array(N,)
            predictive variance

        """

        if not self.is_trained:
            raise ValueError(
                "Calling `bnn.predict()` on an untrained "
                "Bayesian Neural Network 'bnn' is not supported! "
                "Please call `bnn.train()` before calling `bnn.predict()`"
            )

        # Normalize input
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(
                X_test, self.x_mean, self.x_std
            )
        else:
            X_ = X_test

        f_out = []
        theta_noise = []

        for sample in self.samples:
            out = self.compute_network_output(params=sample, input_data=X_)

            f_out.append(out[:, 0])
            theta_noise.append(np.exp(out[:, 1]))

        f_out = np.asarray(f_out)
        theta_noise = np.asarray(theta_noise)

        if return_individual_predictions:
            if self.normalize_output:
                f_out = zero_mean_unit_var_unnormalization(
                    f_out, self.y_mean, self.y_std
                )
                theta_noise *= self.y_std**2
            return f_out, theta_noise

        mean_prediction = np.mean(f_out, axis=0)
        # Total variance
        # v = np.mean(f_out ** 2 + theta_noise, axis=0) - m ** 2
        variance_prediction = np.mean((f_out - mean_prediction) ** 2, axis=0)

        if self.normalize_output:
            mean_prediction = zero_mean_unit_var_unnormalization(
                mean_prediction, self.y_mean, self.y_std
            )
            variance_prediction *= self.y_std ** 2

        return mean_prediction, variance_prediction

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
        if self.normalize_input:
            X = zero_mean_unit_var_unnormalization(self.X, self.x_mean, self.x_std)
            m = self.predict(X)[0]
        else:
            m = self.predict(self.X)[0]

        best_idx = np.argmin(self.y)
        inc = self.X[best_idx]
        inc_value = m[best_idx]

        if self.normalize_input:
            inc = zero_mean_unit_var_unnormalization(inc, self.x_mean, self.x_std)

        if self.normalize_output:
            inc_value = zero_mean_unit_var_unnormalization(inc_value, self.y_mean, self.y_std)

        return inc, inc_value
