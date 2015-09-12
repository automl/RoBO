# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:52:13 2015

@author: aaron
"""
import os
import sys
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
from robo.recommendation.optimize_posterior import optimize_posterior_mean_and_std
from robo.recommendation.incumbent import compute_incumbent


"""
 Experiment to compare the posterior distribution that were obtained by different acquisition function.
 The experiment is in line with Philipp's experiments in his entropy search paper. We draw 40 different test function from a GP with SE kernel, lengthscale=0.1, sn2= 1.0
 in the [0,1]^2 dimensional space. Each acquisition function has 100 iterations to find a good posterior. After each iteration we estimate the current incumbent by
 optimizing the posterior mean + std starting from the best point seen so far.
"""


def main(method):
    for i in range(1, 40):
        task = WithinModelComparison(seed=i + 10)
        #Entropy MC
        try:
            path = "/home/kleinaa/experiments/entropy_search/model_comparison/func_" + str(i) + "/"
            os.makedirs(path)
        except:
            pass

        if method == "entropy_mc":
            save_dir = os.path.join(path, "entropy_mc/")
            kernel = GPy.kern.RBF(2, lengthscale=0.1, variance=1.0)
            model = GPyModel(kernel, optimize=False, noise_variance=1e-3)
            acq = EntropyMC(model, task.X_lower, task.X_upper, optimize_posterior_mean_and_std, Nb=50, Nf=800, Np=100)
            maximizer = Direct(acq, task.X_lower, task.X_upper)
            bo = BayesianOptimization(model=model, acquisition_fkt=acq,
                                      recommendation_strategy=optimize_posterior_mean_and_std,
                                      maximize_fkt=maximizer, task=task, save_dir=save_dir)
            bo.run(100)
        elif method == "entropy_mc_light":
            save_dir = os.path.join(path, "entropy_mc_light/")
            kernel = GPy.kern.RBF(2, lengthscale=0.1, variance=1.0)
            model = GPyModel(kernel, optimize=False, noise_variance=1e-3)
            acq = EntropyMC(model, task.X_lower, task.X_upper, optimize_posterior_mean_and_std, Nb=50, Nf=50, Np=100)
            maximizer = Direct(acq, task.X_lower, task.X_upper)
            bo = BayesianOptimization(model=model, acquisition_fkt=acq,
                                      recommendation_strategy=optimize_posterior_mean_and_std,
                                      maximize_fkt=maximizer, task=task, save_dir=save_dir)
            bo.run(100)

        #EI
        elif method == "ei":
            save_dir = os.path.join(path, "ei/")
            kernel = GPy.kern.RBF(2, lengthscale=0.1, variance=1.0)
            model = GPyModel(kernel, optimize=False, noise_variance=1e-3)
            acq = EI(model, task.X_lower, task.X_upper, compute_incumbent)
            maximizer = Direct(acq, task.X_lower, task.X_upper)
            bo = BayesianOptimization(model=model, acquisition_fkt=acq,
                                      recommendation_strategy=optimize_posterior_mean_and_std,
                                      maximize_fkt=maximizer, task=task, save_dir=save_dir)
            bo.run(100)
        #PI
        elif method == "pi":
            save_dir = os.path.join(path, "pi/")
            kernel = GPy.kern.RBF(2, lengthscale=0.1, variance=1.0)
            model = GPyModel(kernel, optimize=False, noise_variance=1e-3)
            acq = PI(model, task.X_lower, task.X_upper, compute_incumbent)
            maximizer = Direct(acq, task.X_lower, task.X_upper)
            bo = BayesianOptimization(model=model, acquisition_fkt=acq,
                                      recommendation_strategy=optimize_posterior_mean_and_std,
                                      maximize_fkt=maximizer, task=task, save_dir=save_dir)
            bo.run(100)
        #UCB
        elif method == "ucb":
            save_dir = os.path.join(path, "ucb/")
            kernel = GPy.kern.RBF(2, lengthscale=0.1, variance=1.0)
            model = GPyModel(kernel, optimize=False, noise_variance=1e-3)
            acq = UCB(model, task.X_lower, task.X_upper)
            maximizer = Direct(acq, task.X_lower, task.X_upper)
            bo = BayesianOptimization(model=model, acquisition_fkt=acq,
                                      recommendation_strategy=optimize_posterior_mean_and_std,
                                      maximize_fkt=maximizer, task=task, save_dir=save_dir)
            bo.run(100)
        #Entropy
        elif method == "entropy":
            save_dir = os.path.join(path, "entropy/")
            kernel = GPy.kern.RBF(2, lengthscale=0.1, variance=1.0)
            model = GPyModel(kernel, optimize=False, noise_variance=1e-3)
            acq = Entropy(model, task.X_lower, task.X_upper, compute_inc=optimize_posterior_mean_and_std, Nb=50)
            maximizer = Direct(acq, task.X_lower, task.X_upper)
            bo = BayesianOptimization(model=model, acquisition_fkt=acq,
                                      recommendation_strategy=optimize_posterior_mean_and_std,
                                      maximize_fkt=maximizer, task=task, save_dir=save_dir)

            bo.run(100)
if __name__ == '__main__':
    main(sys.argv[1])
