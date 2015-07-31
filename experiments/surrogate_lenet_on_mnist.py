'''
Created on Jul 20, 2015

@author: Aaron Klein
'''

import os
import GPy
import logging

from robo.models.GPyModel import GPyModel
from robo.acquisition.EntropyMC import EntropyMC
from robo.maximizers.cmaes import CMAES
from robo.maximizers.direct import Direct
from robo.maximizers.stochastic_local_search import StochasticLocalSearch
from robo.recommendation.incumbent import compute_incumbent
from robo.task.surrogate_lenet_mnist import SurrogateLeNetMnist, SurrogateEnvLeNetMnist
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.acquisition.EnvEntropySearch import EnvEntropySearch
from robo.acquisition.EI import EI
from robo.solver.env_bayesian_optimization import EnvBayesianOptimization
from robo.recommendation.optimize_posterior import optimize_posterior_mean_and_std, env_optimize_posterior_mean_and_std
from robo.visualization.trajectories import get_incumbents_over_iterations, evaluate_test_performance

from sacred import Experiment


ex = Experiment('lenet_on_mnist')


@ex.config
def default_config():
    num_iterations = 50
    save_dir = "/home/kleinaa/experiments/entropy_search/benchmarks/surrogate_lenet_on_mnist"
    num_restarts = 10
    Nb = 1000
    Nf = 1000
    Np = 100
    method = "EntropyMC"
    run_id = 0
    max_method = "CMAES"
    rec_strategy=None


@ex.named_config
def entropy_config():
    method = "EntropyMC"


@ex.named_config
def env_entropy_config():
    method = "EnvEntropy"


@ex.named_config
def ei_config():
    method = "EI"


@ex.named_config
def direct_config():
    max_method="Direct"


@ex.named_config
def cmaes_config():
    max_method = "CMAES"


@ex.named_config
def stls_config():
    max_method = "StLS"


# @ex.named_config
# def optimize_posterior_config():
#     rec_strategy = "optimize_posterior"
#
#
# @ex.named_config
# def env_optimize_posterior_config():
#     rec_strategy = "env_optimize_posterior"


@ex.automain
def main(max_method, method, num_iterations, save_dir, num_restarts, Nb, Nf, Np, run_id):

    output_dir = os.path.join(save_dir, method + "_" + max_method, "run_" + str(run_id))

    if method == "EntropyMC":
        task = SurrogateLeNetMnist()
        kernel = GPy.kern.Matern52(input_dim=task.n_dims)
        model = GPyModel(kernel, optimize=True, num_restarts=num_restarts)

        acquisition_func = EntropyMC(model, task.X_lower, task.X_upper, optimize_posterior_mean_and_std, Nb=Nb, Nf=Nf, Np=Np)

        if max_method == "CMAES":
            maximizer = CMAES(acquisition_func, task.X_lower, task.X_upper)
        elif max_method == "Direct":
            maximizer = Direct(acquisition_func, task.X_lower, task.X_upper)
        elif max_method == "StLS":
            maximizer = StochasticLocalSearch(acquisition_func, task.X_lower, task.X_upper)

        bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          maximize_fkt=maximizer,
                          task=task,
                          recommendation_strategy=optimize_posterior_mean_and_std,
                          save_dir=output_dir,
                          num_save=1)

        bo.run(num_iterations)

    elif method == "EnvEntropy":
        task = SurrogateEnvLeNetMnist()
        kernel = GPy.kern.Matern52(input_dim=task.n_dims)
        model = GPyModel(kernel, optimize=True, num_restarts=num_restarts)

        cost_kernel = GPy.kern.Matern52(input_dim=task.n_dims)
        cost_model = GPyModel(cost_kernel, optimize=True, num_restarts=num_restarts)

        acquisition_func = EnvEntropySearch(model, cost_model, task.X_lower, task.X_upper, env_optimize_posterior_mean_and_std, task.is_env, Nb=Nb, Nf=Nf, Np=Np)

        if max_method == "CMAES":
            maximizer = CMAES(acquisition_func, task.X_lower, task.X_upper)
        elif max_method == "Direct":
            maximizer = Direct(acquisition_func, task.X_lower, task.X_upper)
        elif max_method == "StLS":
            maximizer = StochasticLocalSearch(acquisition_func, task.X_lower, task.X_upper)

        bo = EnvBayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          cost_model=cost_model,
                          maximize_fkt=maximizer,
                          task=task,
                          recommendation_strategy=env_optimize_posterior_mean_and_std,
                          save_dir=output_dir,
                          num_save=1)

        bo.run(num_iterations)

    if method == "EI":
        task = SurrogateLeNetMnist()
        kernel = GPy.kern.Matern52(input_dim=task.n_dims)
        model = GPyModel(kernel, optimize=True, num_restarts=num_restarts)

        acquisition_func = EI(model, task.X_lower, task.X_upper, compute_incumbent)

        if max_method == "CMAES":
            maximizer = CMAES(acquisition_func, task.X_lower, task.X_upper)
        elif max_method == "Direct":
            maximizer = Direct(acquisition_func, task.X_lower, task.X_upper)
        elif max_method == "StLS":
            maximizer = StochasticLocalSearch(acquisition_func, task.X_lower, task.X_upper)
        bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          maximize_fkt=maximizer,
                          task=task,
                          save_dir=output_dir,
                          num_save=1)

        bo.run(num_iterations)

    logging.info("Compute test performance...")
    iter, inc = get_incumbents_over_iterations(output_dir)
    evaluate_test_performance(task, inc, os.path.join(output_dir, "test_error.npy"))
