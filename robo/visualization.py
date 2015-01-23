import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import numpy as np
from robo import BayesianOptimizationError
from robo.acquisition import Entropy



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
            ax = self.fig.add_subplot(self.nrows, self.ncols, num)
            acq_plot = self.plot_acquisition_fkt( ax, one_dim_min, one_dim_max)
            num+=1
        obj_plot = None
        if obj_method:
            obj_plot = self.fig.add_subplot(self.nrows, self.ncols, num)
            self.objective_fkt = bayesian_opt.objective_fkt
            self.plot_objective_fkt(obj_plot, one_dim_min, one_dim_max)
            num+=1
        if model_method:
            if obj_plot is None:
                obj_plot = self.fig.add_subplot(self.nrows, self.ncols, num)
            self.model = bayesian_opt.model
            self.plot_model(obj_plot, one_dim_min, one_dim_max)
        self.fig.savefig(dest_folder + "/" + prefix +"_iteration.png", format='png')
        self.fig.clf()
        plt.close()
        
    def plot_model(self,  ax, one_dim_min, one_dim_max):
        if hasattr(self.model, "visualize"):
            self.model.visualize(ax, one_dim_min, one_dim_max)
        return ax
    
    def plot_acquisition_fkt(self, ax, one_dim_min, one_dim_max, acquisition_fkt = None, plot_attr={"color":"red"}):
        
        acquisition_fkt = acquisition_fkt or self.acquisition_fkt
        try:
            if isinstance(acquisition_fkt, Entropy):
                self.plot_entropy_acquisition_fkt( ax, one_dim_min, one_dim_max, acquisition_fkt)
            ax.plot(self.plotting_range, acquisition_fkt(self.plotting_range[:,np.newaxis]), **plot_attr)
        except BayesianOptimizationError, e:
            if e.errno ==  BayesianOptimizationError.SINGLE_INPUT_ONLY:
                acq_v =  [ acquisition_fkt(np.array([x])) for x in self.plotting_range[:,np.newaxis] ]
                ax.plot(self.plotting_range, acq_v)
            else:
                raise
        ax.set_xlim(one_dim_min, one_dim_max)
        return ax
    
    def plot_entropy_acquisition_fkt(self, ax, one_dim_min, one_dim_max, acquisition_fkt):
        zb = acquisition_fkt.zb
        pmin = np.exp(acquisition_fkt.logP)
        ax.bar(zb, pmin, width=(one_dim_max - one_dim_min)/(2*zb.shape[0]), color="yellow")
        self.plot_acquisition_fkt(ax, one_dim_min, one_dim_max, acquisition_fkt.sampling_acquisition, {"color":"orange"})
    
    def plot_objective_fkt(self, ax, one_dim_min, one_dim_max):
        
        ax.plot(self.plotting_range, self.objective_fkt(self.plotting_range[:,np.newaxis]), color='b', linestyle="--")
        ax.set_xlim(one_dim_min, one_dim_max)
        ax._min_y, ax._max_y = ax.get_ylim()
        print ax._min_y, ax._max_y
        return ax
    
    def plot_improvement(self, observations):
        pass
        