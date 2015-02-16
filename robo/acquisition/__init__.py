#encoding=utf8
"""
this module contains acquisition functions that should be maximized to find
the minimum of the objective function.

.. class:: AcquisitionFunction

    An acquisition function is a class that gets instatiated with a model 
    and optional additional parameters. It then gets called via a maximizer.

    .. method:: __init__(model, **optional_kwargs)
                
        :param model: A model should have at least the function getCurrentBest() 
                      and predict(X, Z).

    .. method:: __call__(X, Z=None, derivative=False)
               
        :param X: X values, where to evaluate at. It's shape is
        :type X: np.ndarray (N, input_dimension)
        :param Z: instance features to evaluate at. Can be None.
        :param derivative: if a derivative should be calclualted and returnd
        :type derivative: Boolean
        
        :returns:
        
    
    .. method:: update(model)
    
        this method should be called if the model is updated. E.g. the Entropy search needs
        to update its aproximation about P(x=x_min) 
"""

from .LogEI import LogEI
from .PI import PI
from .EI import EI
from .Entropy import Entropy
from .EntropyMC import EntropyMC
from .UCB import UCB














