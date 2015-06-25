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
        elif self.model != None:
            return self.compute(X, derivative)

    def compute(self, X, derivative=False):
        raise NotImplementedError()

    def plot(self, fig, minx, maxx, plot_attr={"color": "red"}, resolution=1000):
        """
            Adds for the acquisition function a subplot to fig. It can create more than one subplot. It's designed for one dimensional objective functions.

            :param fig: the figure on which the subplot will be added
            :type fig: matplotlib.figure.Figure
            :param minx: Lower plotting bound
            :type minx: int
            :param maxx: Upper plotting bound
            :type maxx: int
        """
        n = len(fig.axes)
        for i in range(n):
            fig.axes[i].change_geometry(n + 1, 1, i + 1)
        ax = fig.add_subplot(n + 1, 1, n + 1)
        plotting_range = np.linspace(minx, maxx, num=resolution)
        try:
            ax.plot(plotting_range, self(plotting_range[:, np.newaxis]), **plot_attr)

        except BayesianOptimizationError, e:
            if e.errno == BayesianOptimizationError.SINGLE_INPUT_ONLY:
                acq_v = np.array([self(np.array([x]))[0][0] for x in plotting_range[:, np.newaxis]])
                ax.plot(plotting_range, acq_v, **plot_attr)
            else:
                raise
        ax.set_xlim(minx, maxx)
        ax.set_title(str(self))
        return ax
