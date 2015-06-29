#encoding=utf8

import numpy as np
from robo import BayesianOptimizationError
from robo.models.GPyModelMCMC import GPyModelMCMC


class AcquisitionFunction(object):
    """
    A base class for acquisition functions.
    """
    long_name = ""

    def __str__(self):
        return type(self).__name__ + " (" + self.long_name + ")"

    def __init__(self, model, X_lower, X_upper, **kwargs):
        """
        Initializes the acquisition function

        :param model: Model that captures our current belief of the objective function
        :param X_lower: Lower bound of input space
        :type X_lower: np.ndarray(D, 1)
        :param X_upper: Upper bound of input space
        :type X_upper: np.ndarray(D, 1)

        """
        if isinstance(model, GPyModelMCMC):
            self.mcmc_model = model
            self.model = None
        else:
            self.model = model
            self.mcmc_model = None
        self.X_lower = X_lower
        self.X_upper = X_upper

    def update(self, model):
        """
            This method will be called if the model is updated. E.g. the Entropy search uses it
            to update it's approximation of P(x=x_min)
        """
        if isinstance(model, GPyModelMCMC):
            self.mcmc_model = model
            self.model = None
        else:
            self.model = model
            self.mcmc_model = None

    def __call__(self, X, derivative=False):
        """
            :param X: X values, where the acquisition function should be evaluate. The dimensionality of X is (N, D), with N as the number of points to evaluate
                        at and D is the number of dimensions of one X.
            :type X: np.ndarray (N, D)
            :param derivative: if the derivatives should be computed
            :type derivative: Boolean
            :raises BayesianOptimizationError.NO_DERIVATIVE: if derivative is True and the acquisition function does not allow to compute the gradients
            :rtype: np.ndarray(N, 1)
        """
        if self.mcmc_model != None:
            acq_val = np.zeros([len(self.mcmc_model.models)])
            for i, model in enumerate(self.mcmc_model.models):
                self.model = model
                acq_val[i] = self.compute(X, derivative)
            return np.array([acq_val.sum()])
        elif self.model != None:
            return self.compute(X, derivative)

    def compute(self, X, derivative=False):
        """
            :param X: X values, where the acquisition function should be evaluate. The dimensionality of X is (N, D), with N as the number of points to evaluate
                        at and D is the number of dimensions of one X.
            :type X: np.ndarray (N, D)
            :param derivative: if the derivatives should be computed
            :type derivative: Boolean
            :raises BayesianOptimizationError.NO_DERIVATIVE: if derivative is True and the acquisition function does not allow to compute the gradients
            :rtype: np.ndarray(N, 1)
        """
        raise NotImplementedError()
