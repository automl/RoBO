import os
import random
import errno

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import GPy
import pylab as pb
#pb.ion()
from models import GPyModel 
import numpy as np
from test_functions import branin2, branin
from acquisition import pi_fkt, ucb_fkt
from minimize import DIRECT
here = os.path.abspath(os.path.dirname(__file__))


def bayesian_optimization(objective_fkt, acquisition_fkt, model, minimize_fkt, X_lower, X_upper,  maxN = 10, callback_fkt=lambda model, acq, i:None):
    
    for i in xrange(maxN):
        new_x = minimize_fkt(acquisition_fkt, X_lower, X_upper)
        new_y = objective_fkt(new_x)
        callback_fkt(model, acquisition_fkt, objective_fkt, i)
        model.update(new_x, new_y)




def main():
    obj_samples = 700
    plot_min = -8
    plot_max = 19
    plotting_range = np.linspace(plot_min, plot_max, num=obj_samples)
    second_branin_arg = np.empty((obj_samples,))
    second_branin_arg.fill(12)
    branin_result = branin([plotting_range, second_branin_arg])
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
        fig.savefig("%s/tmp/np_%s.png"%(here, i), format='png')
        fig.clf()
        plt.close() 
        
    kernel = GPy.kern.rbf(input_dim=1, variance=12.3, lengthscale=5.0)
    X_lower = np.array([-8])
    X_upper = np.array([19])
    X = np.empty((1, 1))
    Y = np.empty((1, 1))
    X[0,:] = [random.random() * (X_upper[0] - X_lower[0]) + X_lower[0]];
    objective_fkt= branin2
    Y[0,:] = objective_fkt(X[0,:])
    model = GPyModel(kernel)
    model.train(X,Y)
    acquisition_fkt =  pi_fkt(model)
    bayesian_optimization(objective_fkt, acquisition_fkt, model, DIRECT, X_lower, X_upper, maxN = 10, callback_fkt=_plot_model)
    
    

if __name__ == "__main__":
    
    try:
        os.makedirs("%s/tmp/"%here)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    #from GPy.examples.non_gaussian import student_t_approx
    #student_t_approx(plot=True)
    #plt.show()
    
    main()
