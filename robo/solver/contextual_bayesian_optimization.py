import os
import numpy as np
from robo.acquisition.base import AcquisitionFunction

try:
    import cpickle as pickle
except:
    import pickle
from robo.util.exc import BayesianOptimizationError
from robo import BayesianOptimization
from argparse import ArgumentError

here = os.path.abspath(os.path.dirname(__file__))


class _ContextualAcquisitionFunction(AcquisitionFunction):
    """
    Composite acquisition function for contextual bayesian optimization.
    Calculates the acquisition function by prepending the context to the input.
    """
    long_name = "Contextual Acquisition Function Wrapper"

    def __init__(self, acquisition_fkt, model, X_lower, X_upper, **kwargs):
        """
            Initializes the contextual acquisition function by a base acquisition function.
            The used model must contain the current context.
            :param acquisition_fkt: underlying acquisition function
            """
        super(_ContextualAcquisitionFunction, self).__init__(model, X_lower, X_upper, **kwargs)
        self.acquisition_fkt = acquisition_fkt
        self.Z = None

    def set_context(self, Z):
        """
        Sets the context
        :param Z: The new context vector
        """
        self.Z = Z

    def __call__(self, S, derivative=False, **kwargs):
        # Prepend context to the action variable to obtain X = Z x S
        X = np.concatenate((np.tile(self.Z, (S.shape[0], 1)), S), axis=1)
        return self.acquisition_fkt(X, derivative, **kwargs)

    def update(self, model):
        # Simply forward to the underlying acquisition function
        self.acquisition_fkt.update(model)

    def plot(self, fig, minx, maxx, plot_attr={"color": "red"}, resolution=1000):
        # Simply forward to the underlying acquisition function
        self.acquisition_fkt.plot(fig, minx, maxx, plot_attr, resolution)

    def __str__(self):
        return type(self).__name__ + " (Wraps " + self.acquisition_fkt + ")"


class MeanAcquisitionFunction(AcquisitionFunction):
    """
    Returns the mean of the model

    :param model: A model that implements at least

                 - predict(X)
    :param X_lower: Lower bounds for the search,
            its shape should be 1xD (D = dimension of search space)
    :type X_lower: np.ndarray (1,D)
    :param X_upper: Upper bounds for the search,
            its shape should be 1xD (D = dimension of search space)
    :type X_upper: np.ndarray (1,D)
    """
    long_name = "Mean"

    def __init__(self, model, X_lower, X_upper, **kwargs):
        self.model = model
        self.X_lower = np.array(X_lower)
        self.X_upper = np.array(X_upper)

    def __call__(self, X, derivative=False, **kwargs):
        if derivative:
            raise BayesianOptimizationError(
                BayesianOptimizationError.NO_DERIVATIVE,
                "Mean  does not support derivative calculation until now")
        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            return np.array([[- np.finfo(np.float).max]])
        mean, _ = self.model.predict(X)
        return -mean

    def update(self, model):
        self.model = model


class ContextualBayesianOptimization(BayesianOptimization):
    """
    Class for contextual Bayesian optimization.
    Adds an context function, which obtains a context vector Z before each
    iteration. The objective is then to optimize for this given context.

    For details see
    [Contextual Gaussian Process Bandit Optimization. 2011. Krause, Andreas and Cheng S. Ong.
    Advances in Neural Information Processing Systems 24.]
    """

    def __init__(self, dims_Z=None, dims_S=None, context_fkt=None,
                 acquisition_fkt=None, model=None,
                 maximize_fkt=None, S_lower=None, S_upper=None,
                 objective_fkt=None, save_dir=None, num_save=1):
        """
        Initializes the Contextual Bayesian Optimization.

        :param dims_Z: Dimension of the context vector
        :param dims_S: Dimension of the action vector
        :param context_fkt: Any function for obtaining a context.
                Calling it without arguments must return the vector Z
        :param acquisition_fkt: Any acquisition function
        :param model: A model
        :param maximize_fkt: The function for maximizing the
                acquisition function
        :param S_lower: Lower bounds (tuple of minimums)
        :param S_upper: Upper bounds (tuple of maximums)
        :param objective_fkt: The objective function to execute in each step.
                 Takes two parameters: Context, Action
        :param save_dir: (optional) The directory to save the iterations to
        :param num_save: (optional) A number specifying the n-th
                iteration to be saved (required if save_dir is specified)
        :return:
        """
        # TODO/voegtlel: Saving the pickle won't work here
        if context_fkt is None:
            raise ArgumentError(context_fkt, "Context function missing")
        # Store
        self.S_lower = S_lower
        self.S_upper = S_upper
        self.dims_Z = dims_Z
        self.dims_S = dims_S
        # Total dimensions
        dims = dims_Z + dims_S
        self.Z = None
        # Prepend zeros to the limits
        X_lower = np.concatenate((np.tile(-np.inf, dims_Z), S_lower))
        X_upper = np.concatenate((np.tile(np.inf, dims_Z), S_upper))
        # Wrap the maximize function (removes the context from the maximize
        # function and prepends it)
        maximize_fkt_new = lambda acquisition_fkt, X_lower, X_upper: np.concatenate(
            (self.Z, maximize_fkt(acquisition_fkt, X_lower[dims_Z:], X_upper[dims_Z:])), axis=1)
        self.context_fkt = context_fkt
        # Wrap the acquisition function (prepends the fixed context)
        acquisition_fkt_new = _ContextualAcquisitionFunction(
            acquisition_fkt, model, X_lower, X_upper)
        objective_fkt_new = lambda X: objective_fkt(X[:, :dims_Z], X[:, dims_Z:])
        # For prediction (without exploration)
        self.predict_acquisition_fkt = _ContextualAcquisitionFunction(
            MeanAcquisitionFunction(model, S_lower, S_upper), model, X_lower, X_upper)

        # Forward arguments to super constructor
        super(
            ContextualBayesianOptimization,
            self).__init__(
            acquisition_fkt_new,
            model,
            maximize_fkt_new,
            X_lower,
            X_upper,
            dims,
            objective_fkt_new,
            save_dir,
            num_save)
        self.recommendation_strategy = lambda model, acquisition_fkt: None

    def initialize(self, rng=None):
        self.Z = self.context_fkt()
        if rng is None:
            self.rng = np.random.RandomState(42)
        else:
            self.rng = rng
        # Draw one random configuration
        self.X = np.concatenate(
            (self.Z, np.array([self.rng.uniform(self.S_lower, self.S_upper, self.dims_S)])), 1)
        print "Evaluate randomly chosen candidate %s" % (str(self.X[0]))
        self.Y = self.objective_fkt(self.X)
        print "Configuration achieved a performance of %f " % (self.Y[0])

    def predict(self, Z=None):
        """
        Chooses the optimum of the model (approx. incumbent for a context)
        :param Z: context, may be none (then the context_fkt is questioned)
        :return: the next point where to evaluate as concatenated context x action
        """
        if Z:
            self.Z = Z
        else:
            self.Z = self.context_fkt()
        self.predict_acquisition_fkt.set_context(self.Z)
        # Optimize using the mean
        return self.maximize_fkt(self.predict_acquisition_fkt, self.X_lower, self.X_upper)

    def choose_next(self, X=None, Y=None, Z=None):
        """
        Chooses the next point to evaluate
        :param X: X data
        :param Y: Y data
        :param Z: Z data (if not given, the context_fkt is questioned)
        :return: the next point where to evaluate as concatenated context x action
        """
        # First fetch a context and store it
        if Z:
            self.Z = Z
        else:
            self.Z = self.context_fkt()
        self.acquisition_fkt.set_context(self.Z)
        # Then perform the original optimization
        return super(ContextualBayesianOptimization, self).choose_next(X, Y)
