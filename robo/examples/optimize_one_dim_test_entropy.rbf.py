import os
import random

import matplotlib.pyplot as plt;
import GPy
import pylab as pb
#pb.ion()
from robo import BayesianOptimization
from robo.models import GPyModel 
import numpy as np
from robo.test_functions import one_dim_test as one_dim_test
from robo.acquisition import Entropy
from robo.maximize import  grid_search

here = os.path.abspath(os.path.dirname(__file__))
save_dir = "%s/../tmp/example_optimize_one_dim_test_entropy/" % here
def main(save_dir):
    #
    # Dimension Space where the 
    # objective function can be evaluated 
    #
    
    objective_fkt= one_dim_test
    dims = objective_fkt.dims
    X_lower = objective_fkt.X_lower;
    X_upper = objective_fkt.X_upper;
    
    #
    # Building up the model
    #
    kernel = GPy.kern.RBF(input_dim=dims)    
    model = GPyModel(kernel, optimize=True)

    #
    # creating an acquisition function
    #
    acquisition_fkt = Entropy(model, X_upper= X_upper, X_lower=X_lower)
    #
    # start the main loop
    #
    bo = BayesianOptimization(acquisition_fkt=acquisition_fkt, model=model, maximize_fkt=grid_search, X_lower=X_lower, X_upper=X_upper, dims=dims, objective_fkt=objective_fkt, save_dir=save_dir)
    bo.run(20.0, overwrite=True)
    bo.run(20.0, overwrite=False)

if __name__ == "__main__":
    main(save_dir)