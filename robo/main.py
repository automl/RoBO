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
from acquisition import PI, UCB, Entropy, EI
from maximize import cma, DIRECT, grid_search

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
    fig.savefig("%s/tmp/np_%s.png"%(here, i), format='png')
    fig.clf()
    plt.close()
    

def bayesian_optimization(objective_fkt, acquisition_fkt, model, minimize_fkt, X_lower, X_upper,  maxN = 10):
    for i in xrange(maxN):
        acquisition_fkt.model_changed()

        new_x = minimize_fkt(acquisition_fkt, X_lower, X_upper)
        new_y = objective_fkt(new_x)
        _plot_model(model, acquisition_fkt, objective_fkt, i)
        model.update([new_x], [new_y])

def main():
    #
    # Dimension Space where the 
    # objective function can be evaluated 
    #
    dims = 1
    X_lower = np.array([-8])
    X_upper = np.array([19])
    #initialize the samples
    X = np.empty((1, dims))

    Y = np.empty((1, 1))
    #draw a random sample from the objective function in the
    #dimension space 
    X[0,:] = [random.random() * (X_upper[0] - X_lower[0]) + X_lower[0]];
    objective_fkt= branin2
    Y[0:] = objective_fkt(X[0,:])
    
    #
    # Building up the model
    #    
    kernel = GPy.kern.rbf(input_dim=dims, variance=12.3, lengthscale=5.0)
    model = GPyModel(kernel)
    model.train(X,Y)
    #
    # creating an acquisition function
    #
    acquisition_fkt = PI(model, par=15.01)
    #
    # start the main loop
    #
    bayesian_optimization(objective_fkt, acquisition_fkt, model, grid_search, X_lower, X_upper, maxN = 10)
    
    

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
