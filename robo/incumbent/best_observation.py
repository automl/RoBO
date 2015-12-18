# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 22:40:35 2015

@author: aaron
"""

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
