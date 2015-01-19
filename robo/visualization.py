import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import numpy as np


class Visualization(object):
    def __init__(self, bayesian_opt, new_x, X, Y, dest_folder, prefix="", acq_method = False, obj_method = False, resolution=1000):
        if bayesian_opt.dims > 1 and acq_method:
            raise AttributeError("acquisition function can only be visualized if the objective funktion has only one dimension")
        self.nrows = reduce(lambda x, y : x + 1 if y else x, [0, acq_method, obj_method])
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
            self.plot_acquisition_fkt(num)
            num+=1
        if obj_method:
            self.objective_fkt = bayesian_opt.objective_fkt
            self.plot_objective_fkt(num)
            num+=1
            
        self.fig.savefig(dest_folder + "/" + prefix +"_iteration.png", format='png')
        self.fig.clf()
        plt.close()
        
    def plot_model(self):
        """
        model.m.plot(ax=ax, plot_limits=[plot_min, plot_max])
        xlim_min, xlim_max, ylim_min, ylim_max =  ax.axis()
        ax.set_ylim(min(np.min(branin_result), ylim_min), max(np.max(branin_result), ylim_max))
        """
        pass
    
    def plot_acquisition_fkt(self, num):
        ax = self.fig.add_subplot(self.nrows, self.ncols, num)
        ax.plot(self.plotting_range, self.acquisition_fkt(self.plotting_range[:,np.newaxis]))
        
        """
        c1 = np.reshape(plotting_range, (obj_samples, 1))
        c2 = acquisition_fkt(c1)
        c2 = c2*50 / np.max(c2)
        c2 = np.reshape(c2,(obj_samples,))
        ax.plot(plotting_range,c2, 'r')
        ax.plot(plotting_range, branin_result, 'black')
        fig.savefig("%s/tmp/np_%s.png"%(here, i), format='png')
        fig.clf()
        plt.close()
        """
        
    def plot_objective_fkt(self, num):
        ax = self.fig.add_subplot(self.nrows, self.ncols, num)
        ax.plot(self.plotting_range, self.objective_fkt(self.plotting_range[:,np.newaxis]))
    
    def plot_improvement(self, observations):
        pass
        