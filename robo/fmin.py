'''
Created on Jul 3, 2015

@author: Aaron Klein
'''

import GPy

from robo.models.GPyModel import GPyModel
from robo.models.GPyModelMCMC import GPyModelMCMC
from robo.acquisition.EntropyMC import EntropyMC
from robo.acquisition.Entropy import Entropy
from robo.acquisition.EI import EI
from robo.acquisition.UCB import UCB
from robo.acquisition.PI import PI
from robo.maximizers.maximize import cmaes, direct, stochastic_local_search, grid_search
from robo.recommendation.incumbent import compute_incumbent
from robo.bayesian_optimization import BayesianOptimization


def fmin(objective_fkt, X_lower, X_upper, num_iterations=30, model="GPy", maximizer="direct", kernel="Matern52", acquisition_fkt="EI"):
    dims = X_lower.shape[0]
    assert X_upper.shape[0] == dims

    if kernel == "Matern52":
        k = GPy.kern.Matern52(input_dim=dims)
    elif kernel == "Matern32":
        k = GPy.kern.Matern32(input_dim=dims)
    elif kernel == "RBF":
        k = GPy.kern.RBF(input_dim=dims)
    else:
        print "ERROR: Kernel %s is not a valid kernel!" % (kernel)
        return None

    if model == "GPy":
        m = GPyModel(k, optimize=True, noise_variance=1e-4, num_restarts=10)
    elif model == "GPyMCMC":
        m = GPyModelMCMC(k, optimize=True, noise_variance=1e-4, num_restarts=10)
    else:
        print "ERROR: %s is not a valid model!" % (model)
        return None

    if acquisition_fkt == "EI":
        a = EI(m, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1)
    elif acquisition_fkt == "PI":
        a = PI(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1)
    elif acquisition_fkt == "UCB":
        a = UCB(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent)
    elif acquisition_fkt == "Entropy":
        a = Entropy(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1)
    elif acquisition_fkt == "EntropyMC":
        a = EntropyMC(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1)
    else:
        print "ERROR: %s is not a valid acquisition function!" % (acquisition_fkt)
        return None

    if maximizer == "cmaes":
        max_fkt = cmaes
    elif maximizer == "direct":
        max_fkt = direct
    elif maximizer == "stochastic_local_search":
        max_fkt = stochastic_local_search
    elif maximizer == "grid_search":
        max_fkt = grid_search
    else:
        print "ERROR: %s is not a valid function to maximize the acquisition function!" % (acquisition_fkt)
        return None

    bo = BayesianOptimization(acquisition_fkt=a,
                          model=m,
                          maximize_fkt=max_fkt,
                          X_lower=X_lower,
                          X_upper=X_upper,
                          dims=dims,
                          objective_fkt=objective_fkt,)

    best_x, f_min = bo.run(num_iterations)
    return best_x, f_min
