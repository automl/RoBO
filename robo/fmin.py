'''
Created on Jul 3, 2015

@author: Aaron Klein
'''
import logging
import george
import numpy as np


from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.acquisition.entropy_mc import EntropyMC
from robo.acquisition.entropy import Entropy
from robo.acquisition.ei import EI
from robo.acquisition.lcb import LCB
from robo.acquisition.pi import PI
from robo.maximizers import cmaes, direct, grid_search, stochastic_local_search
from robo.priors.default_priors import DefaultPrior
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.task.base_task import BaseTask

logger = logging.getLogger(__name__)


def fmin(objective_fkt,
        X_lower,
        X_upper,
        num_iterations=30,
        maximizer="direct",
        acquisition_fkt="EI"):

    assert X_upper.shape[0] == X_lower.shape[0]

    class Task(BaseTask):

        def __init__(self, X_lower, X_upper, objective_fkt):
            super(Task, self).__init__(X_lower, X_upper)
            self.objective_function = objective_fkt

    task = Task(X_lower, X_upper, objective_fkt)

    noise = 1.0
    cov_amp = 2

    initial_ls = np.ones([task.n_dims])
    exp_kernel = george.kernels.Matern52Kernel(initial_ls,
                                               ndim=task.n_dims)
    noise_kernel = george.kernels.WhiteKernel(noise, ndim=task.n_dims)
    kernel = cov_amp * (exp_kernel + noise_kernel)

    prior = DefaultPrior(len(kernel))

    model = GaussianProcessMCMC(kernel, prior=prior,
                                n_hypers=20,
                                chain_length=100,
                                burnin_steps=50)

    if acquisition_fkt == "EI":
        a = EI(model, X_upper=task.X_upper, X_lower=task.X_lower)
    elif acquisition_fkt == "PI":
        a = PI(model, X_upper=task.X_upper, X_lower=task.X_lower)
    elif acquisition_fkt == "UCB":
        a = LCB(model, X_upper=task.X_upper, X_lower=task.X_lower)
    elif acquisition_fkt == "Entropy":
        a = Entropy(model, X_upper=task.X_upper, X_lower=task.X_lower)
    elif acquisition_fkt == "EntropyMC":
        a = EntropyMC(model, X_upper=task.X_upper, X_lower=task.X_lower,)
    else:
        logger.error("ERROR: %s is not a"
                    "valid acquisition function!" % (acquisition_fkt))
        return None

    if maximizer == "cmaes":
        max_fkt = cmaes.CMAES(a, task.X_lower, task.X_upper)
    elif maximizer == "direct":
        max_fkt = direct.Direct(a, task.X_lower, task.X_upper)
    elif maximizer == "stochastic_local_search":
        max_fkt = stochastic_local_search.StochasticLocalSearch(a,
                                                    task.X_lower,
                                                    task.X_upper)
    elif maximizer == "grid_search":
        max_fkt = grid_search.GridSearch(a, task.X_lower, task.X_upper)
    else:
        logger.error(
            "ERROR: %s is not a valid function"
            "to maximize the acquisition function!" %
            (acquisition_fkt))
        return None

    bo = BayesianOptimization(acquisition_func=a,
                              model=model,
                              maximize_func=max_fkt,
                              task=task)

    best_x, f_min = bo.run(num_iterations)
    return best_x, f_min
