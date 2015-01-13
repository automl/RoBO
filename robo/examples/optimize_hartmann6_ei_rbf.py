import os
import random
import errno

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import GPy
import pylab as pb
#pb.ion()
from robo import BayesianOptimization
from robo.models import GPyModel 
import numpy as np
from robo.test_functions import hartmann6
from robo.acquisition import EI
from robo.maximize import cma, DIRECT, grid_search
#np.seterr(all='raise')
here = os.path.abspath(os.path.dirname(__file__))
def main():
    #
    # Dimension Space where the 
    # objective function can be evaluated 
    #
    dims = 6
    X_lower = np.array([0.0,0.0,0.0,0.0,0.0,0.0]);
    X_upper = np.array([1.0,1.0,1.0,1.0,1.0,1.0]);
    #initialize the samples
        
    objective_fkt= hartmann6
    
    #
    # Building up the model
    #
    kernel = GPy.kern.RBF(input_dim=dims)    
    model = GPyModel(kernel, optimize=True)
    #
    # creating an acquisition function
    #
    acquisition_fkt = EI(model, X_upper= X_upper, X_lower=X_lower)
    #
    # start the main loop
    #
    bo = BayesianOptimization(acquisition_fkt, model, cma, X_lower, X_upper, dims, objective_fkt)
    bo.run(10)
    

if __name__ == "__main__":
    try:
        os.makedirs("%s/../tmp/"%here)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    #from GPy.examples.non_gaussian import student_t_approx
    #student_t_approx(plot=True)
    #plt.show()
    
    main()
