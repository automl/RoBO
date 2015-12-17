'''
Created on Oct 14, 2015

@author: Aaron Klein
'''

import numpy as np


class BasePrior(object):

    def __init__(self):
        """
        Abstract base class to define the interface for priors
        of GP hyperparameter.
        """
        pass

    def lnprob(self, theta):
        """
        Returns the log probability of theta. Note: theta should
        be on a log scale.
        
        Parameters
        ----------
        theta : (D,) numpy array
            A hyperparameter configuration in log space.
        
        Returns
        -------
        float
            The log probabilty of theta
        """
        pass

    def sample_from_prior(self, n_samples):
        """
        Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        (N, D) np.array
            The samples from the prior.
        """
        pass

    def gradient(self, theta):
        """
        Computes the gradient of the prior with
        respect to theta.

        Parameters
        ----------
        theta : (D,) numpy array
            Hyperparameter configuration in log space

        Returns
        -------
        (D) np.array
            The gradient of the prior at theta.
        """       
        pass
