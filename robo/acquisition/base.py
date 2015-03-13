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
import numpy as np
from robo import BayesianOptimizationError
class AcquisitionFunction(object):
    long_name = ""
    def __str__(self):
        return type(self).__name__ + " (" +self.long_name + ")"
    
    def __init__(self,  model, X_lower, X_upper, **kwargs):
        self.model = model
        self.X_lower = X_lower
        self.X_upper = X_upper
    
    def update(self, model):
        self.model = model
    
    def __call__(self, x):
        raise NotImplementedError()
    
    def plot(self, fig, minx, maxx, plot_attr={"color":"red"}, resolution=1000):
        
        n = len(fig.axes)
        for i in range(n):
            fig.axes[i].change_geometry(n+1, 1, i+1) 
        ax = fig.add_subplot(n+1, 1, n+1) 
        plotting_range = np.linspace(minx, maxx, num=resolution)
        try:
            ax.plot(plotting_range, self(plotting_range[:,np.newaxis]), **plot_attr)
            
        except BayesianOptimizationError, e:
            if e.errno ==  BayesianOptimizationError.SINGLE_INPUT_ONLY:
                acq_v =  np.array([ self(np.array([x]))[0][0] for x in plotting_range[:,np.newaxis] ])
                #if scale:
                #    acq_v = acq_v - acq_v.min() 
                #    acq_v = (scale[1] -scale[0]) * acq_v / acq_v.max() +scale[0]
                
                ax.plot(plotting_range, acq_v, **plot_attr)
            else:
                raise
        ax.set_xlim(minx, maxx)
        ax.set_title(str(self))
        return ax