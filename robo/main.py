import os
import random
import errno
import GPy
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import pylab as pb
#pb.ion()
from models import GPyModel 
import numpy as np
from test_functions import branin2 
from acquisition import pi_fkt, ucb_fkt
from minimize import DIRECT
here = os.path.abspath(os.path.dirname(__file__))


def bayesian_optimization(objective_fkt, acquisition_fkt, model, minimize_fkt, X_lower, X_upper,  maxN = 10, callback_fkt=lambda model, acq, i:None):
    acq = acquisition_fkt(model)
    for i in xrange(maxN):
        new_x = minimize_fkt(acq, X_lower, X_upper)
        new_y = objective_fkt(new_x)
        model.update(new_x, new_y)
        callback_fkt(model, acq, i)

def _plot_model(model, acquisition_fkt, i):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    model.m.plot(ax=ax, plot_limits=[-8, 19])
    obj_samples = 70
    c1 = np.reshape(np.linspace(-8, 19, num=obj_samples), (obj_samples, 1))
    c2 = acquisition_fkt(c1)
    c2 = c2*50 / np.max(c2)
    c1 = np.reshape(c1,(obj_samples,))
    c2 = np.reshape(c2,(obj_samples,))
    ax.plot(c1,c2, 'r')
    plt.close() 
    fig.savefig("%s/tmp/np_%s.png"%(here, i), format='png')
    fig.clf()

def main():
    kernel = GPy.kern.rbf(input_dim=1, variance=10**2, lengthscale=0.2)
    X_lower = np.array([-8])
    X_upper = np.array([19])
    X = np.empty((2, 1))
    X[0,:] = [12.0]
    X[1,:] = [5.0]
    Y = np.empty((2, 1))
    objective_fkt= branin2
    Y[0,:] = objective_fkt(X[0,:])
    Y[1,:] = objective_fkt(X[1,:])
    model = GPyModel(kernel)
    model.train(X,Y)
    acquisition_fkt =  ucb_fkt
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
