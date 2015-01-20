import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import numpy as np



class Visualization(object):
    def __init__(self, bayesian_opt, new_x, X, Y, dest_folder, prefix="", acq_method = False, obj_method = False, model_method = False, resolution=1000):
        if bayesian_opt.dims > 1 and acq_method:
            raise AttributeError("acquisition function can only be visualized if the objective funktion has only one dimension")
        self.nrows = reduce(lambda x, y : x + 1 if y else x, [0, acq_method, obj_method or model_method])
        self.ncols = 1
        self.prefix = prefix
        num = 1
        self.fig = plt.figure()
        one_dim_min = bayesian_opt.X_lower[0]
        one_dim_max = bayesian_opt.X_upper[0]
        if self.ncols:
            self.plotting_range = np.linspace(one_dim_min,one_dim_max, num=resolution)
        if acq_method:
            self.acquisition_fkt = bayesian_opt.acquisition_fkt
            acq_plot, num = self.plot_acquisition_fkt(num, one_dim_min, one_dim_max)
        obj_plot = None
        if obj_method:
            self.objective_fkt = bayesian_opt.objective_fkt
            obj_plot, num = self.plot_objective_fkt(num, one_dim_min, one_dim_max)
        if model_method:
            self.model = bayesian_opt.model
            self.plot_model(num, obj_plot, one_dim_min, one_dim_max)
        self.fig.savefig(dest_folder + "/" + prefix +"_iteration.png", format='png')
        self.fig.clf()
        plt.close()
        
    def plot_model(self, num, ax, one_dim_min, one_dim_max):
        if ax is None:
            ax = self.fig.add_subplot(self.nrows, self.ncols, num)
            num += 1
            
        if hasattr(self.model, "visualize"):
            self.model.visualize(ax, one_dim_min, one_dim_max)
        return ax, num
        
    
    def plot_acquisition_fkt(self, num, one_dim_min, one_dim_max):
        ax = self.fig.add_subplot(self.nrows, self.ncols, num)
        ax.plot(self.plotting_range, self.acquisition_fkt(self.plotting_range[:,np.newaxis]), 'r')
        ax.set_xlim(one_dim_min, one_dim_max)
        num+=1
        return ax, num
        
        
    def plot_objective_fkt(self, num, one_dim_min, one_dim_max):
        ax = self.fig.add_subplot(self.nrows, self.ncols, num)
        ax.plot(self.plotting_range, self.objective_fkt(self.plotting_range[:,np.newaxis]), 'b')
        ax.set_xlim(one_dim_min, one_dim_max)
        num+=1
        return ax, num
    
    def plot_improvement(self, observations):
        pass
        