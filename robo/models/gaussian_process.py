
import logging
import numpy as np
import gpflow


from robo.util import normalization
from robo.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class GaussianProcess(BaseModel):

    def __init__(self, kernel, prior=None,
                 noise=1e-3, use_gradients=False,
                 normalize_output=False,
                 normalize_input=True,
                 lower=None, upper=None, rng=None):
        """
        Interface to the GPflow library. The GP hyperparameter are obtained
        by optimizing the marginal log likelihood.

        Parameters
        ----------
        kernel : GPflow kernel object
            Specifies the kernel that is used for all Gaussian Process
        prior : prior object
            Defines a prior.
        noise : float
            Noise term that is added to the diagonal of the covariance matrix
            for the Cholesky decomposition.
        use_gradients : bool
            Use gradient information to optimize the negative log likelihood
        lower : np.array(D,)
            Lower bound of the input space which is used for the input space normalization
        upper : np.array(D,)
            Upper bound of the input space which is used for the input space normalization
        normalize_output : bool
            Zero mean unit variance normalization of the output values
        normalize_input : bool
            Normalize all inputs to be in [0, 1]. This is important to define good priors for the
            length scales.
        rng: np.random.RandomState
            Random number generator
        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        self.kernel = kernel
        self.gp = None
        self.prior = prior
        self.noise = noise
        self.use_gradients = use_gradients
        self.normalize_output = normalize_output
        self.normalize_input = normalize_input
        self.X = None
        self.y = None
        self.hypers = []
        self.is_trained = False
        self.lower = lower
        self.upper = upper

    @BaseModel._check_shapes_train
    def train(self, X, y, do_optimize=True):
        """

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true the hyperparameters are optimized otherwise
            the default hyperparameters of the kernel are used.
        """

        if self.normalize_input:
            # Normalize input to be in [0, 1]
            self.X, self.lower, self.upper = normalization.zero_one_normalization(X, self.lower, self.upper)
        else:
            self.X = X

        if self.normalize_output:
            # Normalize output to have zero mean and unit standard deviation
            self.y, self.y_mean, self.y_std = normalization.zero_mean_unit_var_normalization(y)
            if self.y_std == 0:
                raise ValueError("Cannot normalize output. All targets have the same value")
        else:
            self.y = y

        self.mean = gpflow.mean_functions.Linear(1, 0)
        self.gp = gpflow.gpr.GPR(self.X, self.y[:, None], kern=self.kernel, mean_function=self.mean)

        if do_optimize:
            self.optimize()

        self.is_trained = True

    def get_noise(self):
        return self.noise

    def optimize(self):
        """
        Optimizes the marginal log likelihood
        """
        self.gp.optimize()


    def predict_variance(self, x1, X2):
        r"""
        Predicts the variance between a test points x1 and a set of points X2 by
           math: \sigma(X_1, X_2) = k_{X_1,X_2} - k_{X_1,X} * (K_{X,X}
                       + \sigma^2*\mathds{I})^-1 * k_{X,X_2})

        Parameters
        ----------
        x1: np.ndarray (1, D)
            First test point
        X2: np.ndarray (N, D)
            Set of test point
        Returns
        ----------
        np.array(N, 1)
            predictive variance between x1 and X2

        """

        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        # if self.normalize_input:
        #     x1_norm, _, _ = normalization.zero_one_normalization(x1, self.lower, self.upper)
        #     X2_norm, _, _ = normalization.zero_one_normalization(X2, self.lower, self.upper)
        # else:
        #     x1_norm = x1
        #     X2_norm = X2

        x_ = np.concatenate((x1, X2))
        _, var = self.predict(x_, full_cov=True)

        var = var[-1, :-1, np.newaxis]

        return var

    @BaseModel._check_shapes_predict
    def predict(self, X_test, full_cov=False, **kwargs):
        """
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        full_cov: bool
            If set to true than the whole covariance matrix between the test points is returned

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,) or np.array(N, N) if full_cov == True
            predictive variance

        """

        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        if self.normalize_input:
            X_test_norm, _, _ = normalization.zero_one_normalization(X_test, self.lower, self.upper)
        else:
            X_test_norm = X_test

        if not full_cov:
            mu, var = self.gp.predict_y(X_test_norm)
        else:
            mu, var = self.gp.predict_f_full_cov(X_test_norm)

        if self.normalize_output:
            mu = normalization.zero_mean_unit_var_unnormalization(mu, self.y_mean, self.y_std)
            var *= self.y_std ** 2


            # Clip negative variances and set them to the smallest
            # positive float value
        if var.shape[0] == 1:
            var = np.clip(var, np.finfo(var.dtype).eps, np.inf)
        else:
            var = np.clip(var, np.finfo(var.dtype).eps, np.inf)
            var[np.where((var < np.finfo(var.dtype).eps) & (var > -np.finfo(var.dtype).eps))] = 0

        return mu, var

    def sample_functions(self, X_test, n_funcs=1):
        """
        Samples F function values from the current posterior at the N
        specified test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        n_funcs: int
            Number of function values that are drawn at each test point.

        Returns
        ----------
        function_samples: np.array(F, N)
            The F function values drawn at the N test points.
        """

        if self.normalize_input:
            X_test_norm, _, _ = normalization.zero_one_normalization(X_test, self.lower, self.upper)
        else:
            X_test_norm = X_test

        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        funcs = self.gp.predict_f_samples(X_test_norm, n_funcs)

        if self.normalize_output:
            funcs = normalization.zero_mean_unit_var_unnormalization(funcs, self.y_mean, self.y_std)

        if len(funcs.shape) == 1:
            return funcs[None, :]
        else:
            return funcs
        
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
        inc, inc_value = super(GaussianProcess, self).get_incumbent()
        if self.normalize_input:
            inc = normalization.zero_one_unnormalization(inc, self.lower, self.upper)

        if self.normalize_output:
            inc_value = normalization.zero_mean_unit_var_unnormalization(inc_value, self.y_mean, self.y_std)

        return inc, inc_value
