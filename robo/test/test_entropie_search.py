import os
import random
import errno
import unittest
import matplotlib; #matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import GPy
import pylab as pb 
import numpy as np
#pb.ion()
from robo.models import GPyModel
from robo.test_functions import branin2, branin
from robo.acquisition import PI, UCB, Entropy, EI
from robo.maximize import cma, DIRECT, grid_search
here = os.path.abspath(os.path.dirname(__file__))

#
# Plotting Stuff.
# It's really ugly until now, because it only fitts to 1D and to the branin function
# where the second domension is 12
#

obj_samples = 700
plot_min = -8
plot_max = 19
plotting_range = np.linspace(plot_min, plot_max, num=obj_samples)

second_branin_arg = np.empty((obj_samples,))
second_branin_arg.fill(12)
branin_arg = np.append(np.reshape(plotting_range, (obj_samples, 1)), np.reshape(second_branin_arg, (obj_samples, 1)), axis=1)
branin_result = branin(branin_arg)
branin_result = branin_result.reshape(branin_result.shape[0],)
def _plot_model(model, acquisition_fkt, objective_fkt, i):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    model.m.plot(ax=ax, plot_limits=[plot_min, plot_max])
    xlim_min, xlim_max, ylim_min, ylim_max =  ax.axis()
    ax.set_ylim(min(np.min(branin_result), ylim_min), max(np.max(branin_result), ylim_max))
    c1 = np.reshape(plotting_range, (obj_samples, 1))
    c2 = acquisition_fkt(c1)
    c2 = c2*50 / np.max(c2)
    c2 = np.reshape(c2,(obj_samples,))
    ax.plot(plotting_range,c2, 'r')
    ax.plot(plotting_range, branin_result, 'black')
    ax.plot()
    ax.plot(acquisition_fkt.zb,np.exp(acquisition_fkt.logP)*100, marker="o", color="#ff00ff", linestyle="");
    ax.plot(acquisition_fkt.zb,acquisition_fkt.logP, marker="h", color="#00a0ff", linestyle="");
    ax.text(0.95, 0.01, str(acquisition_fkt.current_entropy),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='#222222', fontsize=10)
    fig.savefig("%s/../tmp/pmin_%s.png"%(here, i), format='png')
    fig.clf()
    plt.close()
class EmptySampleTestCase(unittest.TestCase):
    def setUp(self):

        dims = 1
        self.num_initial_vals = 3
        self.X_lower = np.array([-8]);#, -8])
        self.X_upper = np.array([19]);#, 19])
        #initialize the samples
        X = np.empty((self.num_initial_vals, dims))
    
        Y = np.empty((self.num_initial_vals, 1))
        self.x_values = [12.0, 4.0, 8.0, 3.0, -8.0, -4.0, 19.0, 0.1, 16.0, 15.6]
        #draw a random sample from the objective function in the
        #dimension space 
        self.objective_fkt= branin2
        for j in range(self.num_initial_vals):
            for i in range(dims):
                X[j,i] = self.x_values[j]#random.random() * (X_upper[i] - X_lower[i]) + X_lower[i];
            Y[j:] = self.objective_fkt( np.array([X[j,:]]))
        
        #
        # Building up the model
        #    
        #old gpy version 
        try:
            self.kernel = GPy.kern.rbf(input_dim=dims, variance=400.0, lengthscale=5.0)
        #gpy version >=0.6
        except AttributeError, e:
            self.kernel = GPy.kern.RBF(input_dim=dims, variance=400.0, lengthscale=5.0)
            
        self.model = GPyModel(self.kernel, optimize=True, noise_variance = 0.002)
        self.model.train(X,Y)
        #
        # creating an acquisition function
        #
        self.acquisition_fkt = Entropy(self.model, self.X_lower, self.X_upper)
        
    def test_pmin(self):
        for i in xrange(self.num_initial_vals, len(self.x_values)):
            
            self.acquisition_fkt.model_changed()
            new_x = np.array(self.x_values[i]).reshape((1,1,))#  grid_search(self.acquisition_fkt, self.X_lower, self.X_upper)
            new_y = self.objective_fkt(new_x)
            _plot_model(self.model, self.acquisition_fkt, self.objective_fkt, i)
            self.model.update(np.array(new_x), np.array(new_y))
            
            
        
if __name__=="__main__":
    unittest.main()