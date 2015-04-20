# encoding=utf8

"""
This module contains the loss functions used in the calculation of the expected information gain.
For the moment only the logloss function is implemented.

    .. method:: __init__(model, X_lower, X_upper, Nb=100, sampling_acquisition=None, sampling_acquisition_kw={"par":0.0}, Np=200, loss_function=None, **kwargs)

        :param logP: Log-probability values.
        :param lmb: Log values of acquisition function at belief points.
        :param lPred: Log of the predictive distribution
        :param args: Additional parameters
        :return:

"""

import numpy as np

def logLoss(logP, lmb, lPred, *args):

    H =   - np.sum(np.multiply(np.exp(logP), (logP + lmb))) # current entropy
    
    
    dHp = - np.sum(np.multiply(np.exp(lPred), np.add(lPred, lmb)), axis=0) - H # @minus? If you change it, change it above in H, too!
    
    return np.array([dHp])