import os
import random

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import GPy
import pylab as pb
#pb.ion()
from robo import BayesianOptimization
from robo.models import GPyModel 
import numpy as np
from robo.test_functions import goldstein_price_fkt as goldstein_price_fkt
from robo.acquisition import PI
from robo.maximize import cma, DIRECT, grid_search

def main(save_dir):
    #
    # Dimension Space where the 
    # objective function can be evaluated 
    #
    
    objective_fkt= goldstein_price_fkt
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
    acquisition_fkt = PI(model, X_upper= X_upper, X_lower=X_lower)
    #
    # start the main loop
    #
    bo = BayesianOptimization(acquisition_fkt=acquisition_fkt, model=model, maximize_fkt=cma, X_lower=X_lower, X_upper=X_upper, dims=dims, objective_fkt=objective_fkt, save_dir=save_dir)
    #bo.run(10.0, overwrite=True)
    bo.run(10.0, overwrite=False)

if __name__ == "__main__":
    here = os.path.abspath(os.path.dirname(__file__))
    save_dir = "%s/../tmp/example_optimize_goldstein_price_pi/"%here
    main(save_dir)