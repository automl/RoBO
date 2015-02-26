import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import numpy as np
from robo import BayesianOptimizationError
from robo.acquisition import Entropy



class Visualization(object):
    def __init__(self, bayesian_opt, new_x, X, Y, dest_folder, prefix="", acq_method = False, obj_method = False, model_method = False, resolution=1000):
        if bayesian_opt.dims > 1 and acq_method:
            raise AttributeError("acquisition function can only be visualized if the objective funktion has only one dimension")
        self.nrows = 0
        if acq_method:
            self.nrows += 1
        if obj_method or model_method:
            self.nrows += 1
        self.ncols = 1
        self.prefix = prefix
        self.num = 1
        self.fig = plt.figure()
        one_dim_min = bayesian_opt.X_lower[0]
        one_dim_max = bayesian_opt.X_upper[0]
        if self.ncols:
            self.plotting_range = np.linspace(one_dim_min,one_dim_max, num=resolution)
        if acq_method:
            self.acquisition_fkt = bayesian_opt.acquisition_fkt
            acq_plot = self.fig.add_subplot(self.nrows, self.ncols, self.num)
            self.num+=1
            self.acquisition_fkt.plot(acq_plot, one_dim_min, one_dim_max)
        obj_plot = None
        if obj_method:
            obj_plot = self.fig.add_subplot(self.nrows, self.ncols, self.num)
            self.num+=1
            self.objective_fkt = bayesian_opt.objective_fkt
            self.plot_objective_fkt(obj_plot, one_dim_min, one_dim_max)
        if model_method:
            if obj_plot is None:
                obj_plot = self.fig.add_subplot(self.nrows, self.ncols, self.num)
            self.model = bayesian_opt.model
            self.plot_model(obj_plot, one_dim_min, one_dim_max)
        self.fig.savefig(dest_folder + "/" + prefix +"_iteration.png", format='png')
        self.fig.clf()
        plt.close()
        
    def plot_model(self,  ax, one_dim_min, one_dim_max):
        if hasattr(self.model, "visualize"):
            self.model.visualize(ax, one_dim_min, one_dim_max)
        _min_y, _max_y = ax.get_ylim()
        if hasattr(ax, "_min_y") and hasattr(ax, "_min_x"):
            ax._min_y = min(_min_y, ax._min_y)
            ax._max_y = max(_max_y, ax._max_y)
            ax.set_ylim(ax._min_y, ax._max_y)
        else:
            ax._min_y = _min_y
            ax._max_y = _max_y
        return ax
    
    
    def plot_objective_fkt(self, ax, one_dim_min, one_dim_max):
        ax.plot(self.plotting_range, self.objective_fkt(self.plotting_range[:,np.newaxis]), color='b', linestyle="--")
        ax.set_xlim(one_dim_min, one_dim_max)
        _min_y, _max_y = ax.get_ylim()
        if hasattr(ax, "_min_y") and hasattr(ax, "_min_x"):
            
            ax._min_y = min(_min_y, ax._min_y)
            ax._max_y = max(_max_y, ax._max_y)
            ax.set_ylim(ax._min_y, ax._max_y)
        else:
            ax._min_y = _min_y
            ax._max_y = _max_y
            ax.set_ylim(ax._min_y, ax._max_y)
        return ax
    
    def plot_improvement(self, observations):
        pass
        