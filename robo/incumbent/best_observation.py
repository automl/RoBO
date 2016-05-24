# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 22:40:35 2015

@author: aaron
"""
from copy import deepcopy
import numpy as np

from robo.incumbent.incumbent_estimation import IncumbentEstimation


class BestObservation(IncumbentEstimation):

    def __init__(self, model, X_lower, X_upper):
        """
        Defines the observed point that leaded to the best function
        value as the incumbent.

        Parameters
        ----------
        model : Model object
            Models the objective function.
        X_lower : (D) numpy array
            Specified the lower bound of the input space. Each entry
            corresponds to one dimension.
        X_upper : (D) numpy array
            Specified the upper bound of the input space. Each entry
            corresponds to one dimension.
        """
        super(BestObservation, self).__init__(model, X_lower, X_upper)

    def estimate_incumbent(self, startpoints=None):
        """
        Estimates the current incumbent. Note: this is just a lookup of
        the observation that has been made so far and thus do not need
        any startpoints.

        Parameters
        ----------
        startpoints : (N, D) numpy array
            In the case of local search, we start form each point a
            separated local search procedure

        Returns
        -------
        np.ndarray(1, D)
            Incumbent
        np.ndarray(1,1)
            Incumbent value
        """

        best = np.argmin(self.model.Y)
        incumbent = self.model.X[best]
        incumbent_value = self.model.Y[best]

        return incumbent[np.newaxis, :], incumbent_value[:, np.newaxis]


class BestProjectedObservation(IncumbentEstimation):

    def __init__(self, model, X_lower, X_upper, is_env):
        """
        Given some observed points in a environmental variable setting, 
        this incumbent estimation strategy returns the observation with the 
        lowest mean prediction on the projected subspace.

        Parameters
        ----------
        model : Model object
            Models the objective function.
        X_lower : (D) numpy array
            Specified the lower bound of the input space. Each entry
            corresponds to one dimension.
        X_upper : (D) numpy array
            Specified the upper bound of the input space. Each entry
            corresponds to one dimension.
        """
        self.is_env = is_env
        super(BestProjectedObservation, self).__init__(model, X_lower, X_upper)

    def estimate_incumbent(self, startpoints=None):
        """
        Estimates the current incumbent by projecting all observed points
        to the projected subspace and return the point with the lowest mean
        prediction.

        Parameters
        ----------
        startpoints : (N, D) numpy array
            In the case of local search, we start form each point a
            separated local search procedure

        Returns
        -------
        np.ndarray(1, D)
            Incumbent
        np.ndarray(1,1)
            Incumbent value
        """
        
        X_ = deepcopy(self.model.X)
        X_[:, self.is_env==1] = self.X_upper[self.is_env==1]
        y = np.zeros([X_.shape[0]])
        for i in range(y.shape[0]):
            y[i] = self.model.predict(X_[i, None, :])[0]
        best = np.argmin(y)
        incumbent = X_[best]
        incumbent_value = y[best]

        return incumbent[np.newaxis, :], np.array([[incumbent_value]])