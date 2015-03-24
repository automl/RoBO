#encoding=utf8

import numpy as np
from robo import BayesianOptimizationError
class AcquisitionFunction(object):
    """
    An acquisition base class.
    
    :param model:
    :param X_lower: Lower bound of input space
    :type X_lower: np.ndarray(D, 1)
    :param X_upper: Upper bound of input space
    :type X_upper: np.ndarray(D, 1)
    """
    long_name = ""
    def __str__(self):
        return type(self).__name__ + " (" +self.long_name + ")"
    
    def __init__(self,  model, X_lower, X_upper, **kwargs):
        
        self.model = model
        self.X_lower = X_lower
        self.X_upper = X_upper
    
    def update(self, model):
        """
            this method will be called if the model is updated. E.g. the Entropy search needs
            to update its aproximation about P(x=x_min)
        """
        self.model = model
    
    def __call__(self, X, derivative=False):
        """ 
            :param X: X values, where to evaluate at. It's shape is of (N, D), where N is the number of points to evaluate at and D is the Dimension of X.
            :type X: np.ndarray (N, D)
            :param derivative: if a derivative
            :type derivative: Boolean
            :raises BayesianOptimizationError.NO_DERIVATIVE: if derivative is True and acquisition function does not support it 
            :returns: np.ndarray(N, 1) 
        """
        raise NotImplementedError()
    
    def plot(self, fig, minx, maxx, plot_attr={"color":"red"}, resolution=1000):
        """
            Adds the acquisition function a subplot to fig. Can create more than one subplot. It's designed for one dimensional objective functions. 
             
            :param fig: the figure where the subplot will be added
            :type fig: matplotlib.figure.Figure
            :param minx: Lower plotting bound
            :type minx: int
            :param maxx: Upper plotting bound
            :type maxx: int
        """
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
                ax.plot(plotting_range, acq_v, **plot_attr)
            else:
                raise
        ax.set_xlim(minx, maxx)
        ax.set_title(str(self))
        return ax