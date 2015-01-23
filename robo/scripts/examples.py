import os
import random
import argparse
import errno

import matplotlib.pyplot as plt;
import GPy
import pylab as pb
import numpy as np

#pb.ion()
from robo import BayesianOptimization
from robo.models import GPyModel 
from robo.test_functions import one_dim_test, branin, hartmann6, hartmann3, goldstein_price_fkt
from robo.acquisition import EI, PI, LogEI, Entropy, UCB
from robo.maximize import grid_search, DIRECT, cma

def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description='Run robo examples', 
                                     prog='robo_examples')
    parser.add_argument('save_dir', metavar="DESTINATION_FOLDER", type=str)
    parser.add_argument('--overwrite', action="store_true", default=False)
    parser.add_argument('-o', '--objective', default=None, type=str,
                        help='Choose your objective function', 
                        dest="objective", choices= ("one_dim_test", "branin", "hartmann6", "hartmann3", "goldstein_price_fkt"))
    parser.add_argument('-a', '--acquisition',  default=None, type=str,
                        help='Choose the acquisition function', 
                        dest="acquisition", choices= ("EI", "PI", "LogEI", "Entropie", "UCB"))
    parser.add_argument('-m', '--model',  default="GPy",  type=str,
                        choices = ("GPy",),
                        help='Choose the model', 
                        dest="model")
    parser.add_argument('-e', '--maximizer',  default="",  type=str,
                        choices = ("grid_search", "DIRECT", "cma"),
                        help='Choose the acquisition maximizer', 
                        dest="maximizer")
    parser.add_argument('-n', '--num',  default="10",  type=int,
                        help='number of evaluations', 
                        dest="n")
    parser.add_argument('--kernel', default="", nargs="+",  type=str,
                        help='Choose a kernel for GP based models', 
                        dest="kernel")
    args = parser.parse_args()
    #
    # Dimension Space where the 
    # objective function can be evaluated 
    #
    
    if args.overwrite == False:
        try:
            bo = BayesianOptimization(save_dir=args.save_dir)
            bo.run(args.n, overwrite=False)
            exit()
        except OSError as exception:
            if exception.errno != errno.ENOENT:
                raise
    exit()
    objective_fkt= one_dim_test
    exit()
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
    acquisition_fkt = EI(model, X_upper= X_upper, X_lower=X_lower)
    #
    # start the main loop
    #
    #bo = BayesianOptimization(acquisition_fkt=acquisition_fkt, model=model, maximize_fkt=grid_search, X_lower=X_lower, X_upper=X_upper, dims=dims, objective_fkt=objective_fkt, save_dir=save_dir)
    #bo.run(20.0, overwrite=True)
    #bo.run(20.0, overwrite=False)

if __name__ == "__main__":
    main(save_dir)