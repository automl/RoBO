import os
import random
import errno

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import GPy
import pylab as pb
#pb.ion()
from robo import bayesian_optimization_main
from robo.models import GPyModel 
import numpy as np
from robo.test_functions import goldstein_price_fkt as goldstein_price_fkt
from robo.acquisition import PI
from robo.maximize import cma, DIRECT, grid_search
#np.seterr(all='raise')
here = os.path.abspath(os.path.dirname(__file__))

#
# Plotting Stuff.
# It's really ugly until now, because it only fitts to 1D and to the branin function
# where the second domension is 12
#

#obj_samples = 700
#plot_min = -8
#plot_max = 19
#plotting_range = np.linspace(plot_min, plot_max, num=obj_samples)

#second_branin_arg = np.empty((obj_samples,))
#second_branin_arg.fill(12)
#branin_arg = np.append(np.reshape(plotting_range, (obj_samples, 1)), np.reshape(second_branin_arg, (obj_samples, 1)), axis=1)
#branin_result = branin(branin_arg)
#branin_result = branin_result.reshape(branin_result.shape[0],)

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

def main():
    #
    # Dimension Space where the 
    # objective function can be evaluated 
    #
    
    objective_fkt= goldstein_price_fkt
    dims = objective_fkt.dims
    X_lower = objective_fkt.X_lower;
    X_upper = objective_fkt.X_upper;
    #initialize the samples
    X = np.empty((1, dims))
    Y = np.empty((1, 1))
    #draw a random sample from the objective function in the
    #dimension space 
    for i in range(dims):
        X[0,i] = random.random() * (X_upper[i] - X_lower[i]) + X_lower[i];
        
    Y[0:] = objective_fkt(np.array([X[0,:]]))
    
    #
    # Building up the model
    #
    kernel = GPy.kern.RBF(input_dim=dims)    
    model = GPyModel(kernel, optimize=True)
    model.train(X,Y)
    #
    # creating an acquisition function
    #
    acquisition_fkt = PI(model, X_upper= X_upper, X_lower=X_lower)
    #
    # start the main loop
    #
    print "-"*50 +"\n- " + str(bayesian_optimization_main(objective_fkt, acquisition_fkt, model, cma, X_lower, X_upper, maxN = 60)) + "-"*50
    
    

if __name__ == "__main__":
    try:
        os.makedirs("%s/../tmp/"%here)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    main()