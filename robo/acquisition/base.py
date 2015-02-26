#encoding=utf8
"""
this module contains acquisition functions that should be maximized to find
the minimum of the objective function.

.. class:: AcquisitionFunction

    An acquisition function is a class that gets instatiated with a model 
    and optional additional parameters. It then gets called via a maximizer.

    .. method:: __init__(model, X_lower, X_upper **optional_kwargs)
                
        :param model: A model should have at least the function getCurrentBest() 
                      and predict(X, Z).

    .. method:: __call__(X, Z=None, derivative=False)
               
        :param X: X values, where to evaluate at. It's shape is of (N, D), where N is the number of points to evaluate at and D is the Dimension of X.
        :type X: np.ndarray (N, input_dimension)
        :param Z: instance features to evaluate at. Can be None.
        :param derivative: if a derivative should be calclualted and returnd
        :type derivative: Boolean
        
        :returns:
        
    
    .. method:: update(model)
    
        this method should be called if the model is updated. E.g. the Entropy search needs
        to update its aproximation about P(x=x_min) 
"""

class AcquisitionFunction(object):
    def __init__(self,  model, X_lower, X_upper, **kwargs):
        self.model = model
        self.X_lower = X_lower
        self.X_upper = X_upper
    
    def update(self, model):
        self.model = model
    
    def __call__(self, x):
        raise NotImplementedError()