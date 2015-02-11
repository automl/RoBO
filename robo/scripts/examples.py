import os
import random
random.seed(13)
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
from robo.acquisition import EI, PI, LogEI, Entropy, UCB, EntropyMC
from robo.maximize import grid_search, DIRECT, cma, sample_optimizer

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
                        dest="acquisition", choices= ("EI", "PI", "LogEI", "Entropy", "EntropyMC", "UCB"))
    
    parser.add_argument('-p', '--acquisition-parameter',  default=[], type=str, nargs="*",
                        help='Choose the acquisition function parameters', 
                        dest="acquisition_parameters")
    
    parser.add_argument('-m', '--model',  default="GPy",  type=str,
                        choices = ("GPy",),
                        help='Choose the model', 
                        dest="model")
    
    parser.add_argument('-e', '--maximizer',  default="",  type=str,
                        choices = ("grid_search", "DIRECT", "cma", "sample_optimizer"),
                        help='Choose the acquisition maximizer', 
                        dest="maximizer")
    
    parser.add_argument('-n', '--num',  default="10",  type=int,
                        help='number of evaluations', 
                        dest="n")
    
    parser.add_argument('--without-noise', help='disable noise', 
                        dest="with_noise", default=True, action="store_false" )
    
    parser.add_argument('--kernel', default="RBF", nargs="+",  type=str,
                        help='Choose a kernel for GP based models',
                        choices= ("RBF", ),
                        dest="kernel")
    
    
    args = parser.parse_args()
    
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
            
    if args.objective == "one_dim_test":
        objective_fkt= one_dim_test
    elif args.objective == "branin":
        objective_fkt= branin
    elif args.objective == "hartmann6":
        objective_fkt= hartmann6
    elif args.objective == "hartmann3":
        objective_fkt= hartmann3
    elif args.objective == "goldstein_price_fkt":
        objective_fkt= goldstein_price_fkt
    
    dims = objective_fkt.dims
    X_lower = objective_fkt.X_lower;
    X_upper = objective_fkt.X_upper;
    
    #
    # Building up the model
    #
    if args.model in ("GPy",) and args.kernel == "RBF":
        kernel = GPy.kern.RBF(input_dim=dims)
    
    model_kwargs = {}
    if not args.with_noise:
        model_kwargs["noise_variance"] = 1e-3
    if args.model == "GPy":
        model = GPyModel(kernel, optimize=True, **model_kwargs)

    #
    # creating an acquisition function
    #
    acquisition_kwargs = {}
    if args.acquisition == "EI":
        if len(args.acquisition_parameters): 
            acquisition_kwargs["par"] = float(args.acquisition_parameters[0])
        acquisition_fkt = EI(model, X_upper= X_upper, X_lower=X_lower, **acquisition_kwargs)
        
    elif args.acquisition == "PI":
        if len(args.acquisition_parameters): 
            acquisition_kwargs["par"] = float(args.acquisition_parameters[0])
        acquisition_fkt = PI(model, X_upper= X_upper, X_lower=X_lower,  **acquisition_kwargs)
        
    elif args.acquisition == "LogEI":
        if len(args.acquisition_parameters): 
            acquisition_kwargs["par"] = float(args.acquisition_parameters[0])
        acquisition_fkt = LogEI(model, X_upper= X_upper, X_lower=X_lower,  **acquisition_kwargs)
        
    elif args.acquisition == "Entropy": 
        acquisition_fkt = Entropy(model, X_upper= X_upper, X_lower=X_lower,  **acquisition_kwargs)
        
    if args.acquisition == "EntropyMC":
        acquisition_fkt = EntropyMC(model, X_upper= X_upper, X_lower=X_lower,  **acquisition_kwargs)
        
    elif args.acquisition == "UCB":
        if len(args.acquisition_parameters): 
            acquisition_kwargs["par"] = float(args.acquisition_parameters[0])
        acquisition_fkt = UCB(model, X_upper= X_upper, X_lower=X_lower**acquisition_kwargs)

    if args.maximizer == "grid_search":
        maximize_fkt = grid_search
        
    elif args.maximizer == "DIRECT":
        maximize_fkt = DIRECT
        
    elif args.maximizer == "cma":
        maximize_fkt = cma
        
    elif args.maximizer == "sample_optimizer":
        maximize_fkt = sample_optimizer
        
    #
    # start the main loop
    #
    bo = BayesianOptimization(acquisition_fkt=acquisition_fkt, model=model, maximize_fkt=maximize_fkt, X_lower=X_lower, X_upper=X_upper, dims=dims, objective_fkt=objective_fkt, save_dir=args.save_dir)
    bo.run(args.n, overwrite=args.overwrite)
    #bo.run(20.0, overwrite=False)

if __name__ == "__main__":
    main()
