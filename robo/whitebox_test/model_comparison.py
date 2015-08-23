# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:52:13 2015

@author: aaron
"""
import os
import GPy

from robo.maximizers.direct import Direct
from robo.acquisition.entropy_mc import EntropyMC
from robo.acquisition.entropy import Entropy
from robo.acquisition.ei import EI
from robo.acquisition.pi import PI
from robo.acquisition.ucb import UCB
from robo.models.gpy_model import GPyModel
from robo.task.within_model_comparison import WithinModelComparison
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.acquisition.entropy_mc import EntropyMC
from robo.recommendation.optimize_posterior import optimize_posterior_mean_and_std
from robo.recommendation.incumbent import compute_incumbent

for i in range(1, 20):
        try:
            task = WithinModelComparison(seed=i+10)
            #Entropy MC
            try:
                os.makedirs("/home/aaron/experiment/model_comparison/" + str(i) + "/")
            except:
                pass
            save_dir = "/home/aaron/experiment/model_comparison/" + str(i) + "/entropy_mc/"
            kernel = GPy.kern.RBF(2, lengthscale=0.1, variance=1.0)
            model = GPyModel(kernel, optimize=False, noise_variance=1e-3)
            acq = EntropyMC(model, task.X_lower, task.X_upper, optimize_posterior_mean_and_std, Nb=50, Nf=100, Np=40)
            maximizer = Direct(acq, task.X_lower, task.X_upper)
            bo = BayesianOptimization(model=model, acquisition_fkt=acq, 
                                      recommendation_strategy=optimize_posterior_mean_and_std,                            
                                      maximize_fkt=maximizer, task=task, save_dir=save_dir)
            bo.run(40)    
            #EI
            save_dir = "/home/aaron/experiment/model_comparison/" + str(i) + "/ei/"
            kernel = GPy.kern.RBF(2, lengthscale=0.1, variance=1.0)
            model = GPyModel(kernel, optimize=False, noise_variance=1e-3)
            acq = EI(model, task.X_lower, task.X_upper, compute_incumbent)
            maximizer = Direct(acq, task.X_lower, task.X_upper)
            bo = BayesianOptimization(model=model, acquisition_fkt=acq, 
                                      recommendation_strategy=optimize_posterior_mean_and_std,                            
                                      maximize_fkt=maximizer, task=task, save_dir=save_dir)
            bo.run(40) 
            #PI
            save_dir = "/home/aaron/experiment/model_comparison/" + str(i) + "/pi/"
            kernel = GPy.kern.RBF(2, lengthscale=0.1, variance=1.0)
            model = GPyModel(kernel, optimize=False, noise_variance=1e-3)
            acq = PI(model, task.X_lower, task.X_upper, compute_incumbent)
            maximizer = Direct(acq, task.X_lower, task.X_upper)
            bo = BayesianOptimization(model=model, acquisition_fkt=acq, 
                                      recommendation_strategy=optimize_posterior_mean_and_std,                            
                                      maximize_fkt=maximizer, task=task, save_dir=save_dir)
            bo.run(40) 
            #UCB
            save_dir = "/home/aaron/experiment/model_comparison/" + str(i) + "/ucb/"
            kernel = GPy.kern.RBF(2, lengthscale=0.1, variance=1.0)
            model = GPyModel(kernel, optimize=False, noise_variance=1e-3)
            acq = UCB(model, task.X_lower, task.X_upper)
            maximizer = Direct(acq, task.X_lower, task.X_upper)
            bo = BayesianOptimization(model=model, acquisition_fkt=acq, 
                                      recommendation_strategy=optimize_posterior_mean_and_std,                            
                                      maximize_fkt=maximizer, task=task, save_dir=save_dir)
            bo.run(40)
            #Entropy
            save_dir = "/home/aaron/experiment/model_comparison/" + str(i) + "/entropy/"
            kernel = GPy.kern.RBF(2, lengthscale=0.1, variance=1.0)
            model = GPyModel(kernel, optimize=False, noise_variance=1e-3)
            acq = Entropy(model, task.X_lower, task.X_upper, compute_inc=optimize_posterior_mean_and_std, Nb=50)
            maximizer = Direct(acq, task.X_lower, task.X_upper)
            bo = BayesianOptimization(model=model, acquisition_fkt=acq, 
                                      recommendation_strategy=optimize_posterior_mean_and_std,                            
                                      maximize_fkt=maximizer, task=task, save_dir=save_dir)
        
            bo.run(40) 
        except:
            continue
  
    
