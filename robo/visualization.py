
import numpy as np
from robo import BayesianOptimizationError
from robo.acquisition import Entropy

class Visualization(object):
    def __init__(self, bayesian_opt, new_x, X, Y, dest_folder=None, prefix="", show_acq_method = False, show_obj_method = False, show_model_method = False, resolution=1000, interactive=False):
        if dest_folder is not None:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt;
            interactive=False
        if interactive:
            import matplotlib; matplotlib.use('GTKAgg')
            import matplotlib.pyplot as plt;
            plt.ion()
            
        if bayesian_opt.dims > 1 and show_acq_method:
            raise AttributeError("acquisition function can only be visualized if the objective funktion has only one dimension")
        if bayesian_opt.dims > 1 and show_obj_method:
            raise AttributeError("objective function can only be visualized if the objective funktion has only one dimension")
        if bayesian_opt.dims > 1 and show_model_method:
            raise AttributeError("model can only be visualized if the objective funktion has only one dimension")
      
        self.prefix = prefix
        self.fig = plt.figure()
        self.new_x = new_x
        self.obj_plot_min_y = None
        self.obj_plot_max_y = None
        one_dim_min = bayesian_opt.X_lower[0]
        one_dim_max = bayesian_opt.X_upper[0]
        self.num_subplots = 0
        
        self.plotting_range = np.linspace(one_dim_min,one_dim_max, num=resolution)
        if show_acq_method:
            self.acquisition_fkt = bayesian_opt.acquisition_fkt
            self.acquisition_fkt.plot(self.fig, one_dim_min, one_dim_max)
        insert_last = False
        if show_obj_method:
            self.objective_fkt = bayesian_opt.objective_fkt
            self.plot_objective_fkt(self.fig, one_dim_min, one_dim_max)
            insert_last = True
        if show_model_method:
            self.model = bayesian_opt.model
            self.plot_model(self.fig, one_dim_min, one_dim_max, insert_last)
        if not interactive:
            self.fig.savefig(dest_folder + "/" + prefix +"_iteration.png", format='png')
            self.fig.clf()
            plt.close()
        else:
            plt.show(block=True)
        
    def plot_model(self, fig, one_dim_min, one_dim_max, insert_last=False):
        ax = fig.axes[-1]
        if not insert_last:
            n = len(fig.axes)
            for i in range(n):
                fig.axes[i].change_geometry(n+1, 1, i+1) 
            ax = fig.add_subplot(n+1, 1, n+1) 
        if hasattr(self.model, "visualize"):
            self.model.visualize(ax, one_dim_min, one_dim_max)
        _min_y, _max_y = ax.get_ylim()
        mu, var = self.model.predict(self.new_x)
        ax.plot(self.new_x[0], mu[0], "r." , markeredgewidth=5.0)
        if self.obj_plot_min_y is not  None and self.obj_plot_max_y is not None:
            self.obj_plot_min_y = min(_min_y, self.obj_plot_min_y)
            self.obj_plot_max_y = max(_max_y, self.obj_plot_max_y)
            ax.set_ylim(self.obj_plot_min_y, self.obj_plot_max_y)
        else:
            self.obj_plot_min_y = _min_y
            self.obj_plot_max_y = _max_y
        return ax
    
    
    def plot_objective_fkt(self, fig, one_dim_min, one_dim_max):
        n = len(fig.axes)
        for i in range(n):
            fig.axes[i].change_geometry(n+1, 1, i+1) 
        ax = fig.add_subplot(n+1, 1, n+1) 
        ax.plot(self.plotting_range, self.objective_fkt(self.plotting_range[:,np.newaxis]), color='b', linestyle="--")
        ax.set_xlim(one_dim_min, one_dim_max)
        _min_y, _max_y = ax.get_ylim()
        if self.obj_plot_min_y is not  None and self.obj_plot_max_y is not None:
            self.obj_plot_min_y = min(_min_y, self.obj_plot_min_y)
            self.obj_plot_max_y = max(_max_y, self.obj_plot_max_y)
            ax.set_ylim(self.obj_plot_min_y, self.obj_plot_max_y)
        else:
            self.obj_plot_min_y = _min_y
            self.obj_plot_max_y = _max_y
            ax.set_ylim(self.obj_plot_min_y, self.obj_plot_max_y)
        
        return ax
    
  
        