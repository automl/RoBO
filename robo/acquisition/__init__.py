    #encoding=utf8
"""
this module contains acquisition functions that have high values
where the objective function is low.


.. class:: AcquisitionFunction

    An acquisition function is a class that gets instatiated with a model 
    and optional additional parameters. It then gets called via a maximizer.

    .. method:: __init__(model, **optional_kwargs)
                
        :param model: A model should have at least the function getCurrentBest() 
                      and predict(X, Z)

    .. method:: __call__(X, Z=None)
               
        :param X: X values, where to evaluate the acquisition function 
        :param Z: instance features to evaluate at. Could be None.
    
    .. method:: update(model)
    
        this method should be called if the model is updated. The Entropy search needs
        to update its aproximation about P(x=x_min) 
"""

from .LogEI import LogEI
from .PI import PI
from .EI import EI
from .Entropy import Entropy
from .EntropyMC import EntropyMC
class UCB(object):
    def __init__(self, model, par=1.0, **kwargs):
        self.model = model
        self.par = par
    def __call__(self, X, Z=None, **kwargs):
        mean, var = self.model.predict(X, Z)
        return -mean + self.par * var
    def update(self, model):
        self.model = model













