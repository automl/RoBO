'''
Created on Jul 3, 2015

@author: Aaron Klein
'''

import GPy

from robo.models import gpy_model.GPyModel
from robo.models.GPyModelMCMC import GPyModelMCMC
from robo.acquisition.EntropyMC import EntropyMC
from robo.acquisition.Entropy import Entropy
from robo.acquisition.EI import EI
from robo.acquisition.UCB import UCB
from robo.acquisition.PI import PI
from robo.maximizers import cmaes, direct, grid_search, stochastic_local_search
from robo.recommendation.incumbent import compute_incumbent
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.task.base_task import BaseTask


def fmin(objective_fkt, X_lower, X_upper, num_iterations=30, model="GPy", maximizer="direct", kernel="Matern52", acquisition_fkt="EI"):

    assert X_upper.shape[0] == X_lower.shape[0]

    class Task(BaseTask):
        def __init__(self):
            self.X_lower = X_lower
            self.X_upper = X_upper
            self.n_dims = X_lower.shape[0]
            self.objective_function = objective_fkt

    task = Task()

    if kernel == "Matern52":
        k = GPy.kern.Matern52(input_dim=task.n_dims)
    elif kernel == "Matern32":
        k = GPy.kern.Matern32(input_dim=task.n_dims)
    elif kernel == "RBF":
        k = GPy.kern.RBF(input_dim=task.n_dims)
    else:
        print "ERROR: Kernel %s is not a valid kernel!" % (kernel)
        return None

    if model == "GPy":
        m = gpy_model(k, optimize=True, noise_variance=1e-4, num_restarts=10)
    elif model == "GPyMCMC":
        m = GPyModelMCMC(k, optimize=True, noise_variance=1e-4, num_restarts=10)
    else:
        print "ERROR: %s is not a valid model!" % (model)
        return None

    if acquisition_fkt == "EI":
        a = EI(m, X_upper=task.X_upper, X_lower=task.X_lower, compute_incumbent=compute_incumbent, par=0.1)
    elif acquisition_fkt == "PI":
        a = PI(model, X_upper=task.X_upper, X_lower=task.X_lower, compute_incumbent=compute_incumbent, par=0.1)
    elif acquisition_fkt == "UCB":
        a = UCB(model, X_upper=task.X_upper, X_lower=task.X_lower, compute_incumbent=compute_incumbent)
    elif acquisition_fkt == "Entropy":
        a = Entropy(model, X_upper=task.X_upper, X_lower=task.X_lower, compute_incumbent=compute_incumbent, par=0.1)
    elif acquisition_fkt == "EntropyMC":
        a = EntropyMC(model, X_upper=task.X_upper, X_lower=task.X_lower, compute_incumbent=compute_incumbent, par=0.1)
    else:
        print "ERROR: %s is not a valid acquisition function!" % (acquisition_fkt)
        return None

    if maximizer == "cmaes":
        max_fkt = cmaes.CMAES(a, task.X_lower, task.X_upper)
    elif maximizer == "direct":
        max_fkt = direct.Direct(a, task.X_lower, task.X_upper)
    elif maximizer == "stochastic_local_search":
        max_fkt = stochastic_local_search.StochasticLocalSearch(a, task.X_lower, task.X_upper)
    elif maximizer == "grid_search":
        max_fkt = grid_search.GridSearch(a, task.X_lower, task.X_upper)
    else:
        print "ERROR: %s is not a valid function to maximize the acquisition function!" % (acquisition_fkt)
        return None

    bo = BayesianOptimization(acquisition_fkt=a,
                          model=m,
                          maximize_fkt=max_fkt,
                          task=task)

    best_x, f_min = bo.run(num_iterations)
    return best_x, f_min
