'''
Created on Aug 11, 2015

@author: Aaron Klein
'''
import logging
import numpy as np

from copy import deepcopy
from robo.acquisition.base import AcquisitionFunction

logger = logging.getLogger(__name__)

class IntegratedAcquisition(AcquisitionFunction):
    '''
        Meta acquisition function that allows to marginalise the acquisition over different GPs.
    '''

    def __init__(self, model, acquisition_func, cost_model=None):

        self.model = model

        # Save also the cost model if the acquisition function needs it
        if cost_model is not None:

            self.cost_model = cost_model

        # Keep for each model an extra acquisition function module
        self.estimators = []
        for _ in range(self.model.n_hypers):
            # Copy the acquisition function for this model
            estimator = deepcopy(acquisition_func)
            estimator.model = None
            if cost_model is not None:
                estimator.cost_model = None
            self.estimators.append(estimator)

    def update(self, model, cost_model=None):
        self.model = model
        if cost_model is not None:
            self.cost_model = cost_model
        for i in range(self.model.n_hypers):
            if cost_model is not None:
                self.estimators[i].update(self.model.models[i], self.cost_model.models[i],)
            else:
                self.estimators[i].update(self.model.models[i])

    def compute(self, X, derivative=False):
        acquisition_values = np.zeros([self.model.n_hypers])

        # Integrate over the acquisition values
        for i in range(self.model.n_hypers):
            acquisition_values[i] = self.estimators[i].compute(X, derivative)
        return np.array([[acquisition_values.mean()]])
