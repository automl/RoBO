import logging
import numpy as np

from copy import deepcopy

from robo.acquisition_functions.base_acquisition import BaseAcquisitionFunction

logger = logging.getLogger(__name__)


class MarginalizationGPMCMC(BaseAcquisitionFunction):
    def __init__(self, acquisition_func):
        """
        Meta acquisition_functions function that allows to marginalise the
        acquisition_functions function over GP hyperparameters.

        Parameters
        ----------

        acquisition_func: BaseAcquisitionFunction object
            The acquisition_functions function that will be integrated.
        """
        self.acquisition_func = acquisition_func
        self.model = acquisition_func.model

        # Save also the cost model if the acquisition_functions function needs it
        if hasattr(acquisition_func, "cost_model"):
            self.cost_model = acquisition_func.cost_model
        else:
            self.cost_model = None

        # Keep for each model an extra acquisition_functions function module
        self.estimators = []
        for i in range(len(self.model.models)):
            # Copy the acquisition_functions function for this model
            estimator = deepcopy(acquisition_func)
            if len(self.model.models) == 0:
                estimator.model = None
            else:
                estimator.model = self.model.models[i]

            if self.cost_model is not None:
                if len(self.cost_model.models) == 0:
                    estimator.model = None
                else:
                    estimator.model = self.cost_model.models[i]
            self.estimators.append(estimator)

    def update(self, model, cost_model=None, **kwargs):
        """
        Updates each acquisition_functions function object if the models
        have changed

        Parameters
        ----------
        model: Model object
            The model of the objective function, it has to be an instance of
            GaussianProcessMCMC.
        cost_model: Model object
            If the acquisition_functions function also takes the cost into account, we
            have to specify here the model for the cost function. cost_model
            has to be an instance of GaussianProcessMCMC.
        """
        if len(self.estimators) == 0:
            for i in range(len(self.model.models)):
                # Copy the acquisition_functions function for this model
                estimator = deepcopy(self.acquisition_func)
                if len(self.model.models) == 0:
                    estimator.model = None
                else:
                    estimator.model = self.model.models[i]

                if self.cost_model is not None:
                    if len(self.cost_model.models) == 0:
                        estimator.model = None
                    else:
                        estimator.model = self.cost_model.models[i]
                self.estimators.append(estimator)

        self.model = model
        if cost_model is not None:
            self.cost_model = cost_model
        for i in range(len(self.model.models)):

            if cost_model is not None:
                self.estimators[i].update(self.model.models[i],
                                          self.cost_model.models[i],
                                          **kwargs)
            else:
                self.estimators[i].update(self.model.models[i], **kwargs)

    def compute(self, X_test, derivative=False):
        """
        Integrates the acquisition_functions function over the GP's hyperparameters by
        averaging the acquisition_functions value for X of each hyperparameter sample.

        Parameters
        ----------
        X_test: np.ndarray(N, D), The input point where the acquisition_functions function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            If is set to true also the derivative of the acquisition_functions
            function at X is returned

        Returns
        -------
        np.ndarray(1,1)
            Integrated acquisition_functions value of X
        np.ndarray(1,D)
            Derivative of the acquisition_functions value at X (only if derivative=True)
        """
        acquisition_values = np.zeros([len(self.model.models), X_test.shape[0]])

        # Integrate over the acquisition_functions values
        for i in range(len(self.model.models)):
            acquisition_values[i] = self.estimators[i].compute(X_test, derivative=derivative)

        return acquisition_values.mean(axis=0)
