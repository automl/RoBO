'''
Created on Jul 20, 2015

@author: Aaron Klein
'''

import os
import GPy
import logging

from robo.models.GPyModel import GPyModel
from robo.models.hmc_gp import HMCGP
from robo.models.GPyModelMCMC import GPyModelMCMC
from robo.acquisition.EntropyMC import EntropyMC
from robo.maximizers.cmaes import CMAES
from robo.maximizers.direct import Direct
from robo.maximizers.stochastic_local_search import StochasticLocalSearch
from robo.recommendation.incumbent import compute_incumbent
from robo.task.lenet_mnist import LeNetMnist, EnvLeNetMnist
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.acquisition.EnvEntropySearch import EnvEntropySearch
from robo.acquisition.EI import EI
from robo.solver.env_bayesian_optimization import EnvBayesianOptimization
from robo.recommendation.optimize_posterior import optimize_posterior_mean_and_std, env_optimize_posterior_mean_and_std
from robo.visualization.trajectories import get_incumbents_over_iterations, evaluate_test_performance
from robo.acquisition.mcmc_entropy import MarginalisingHyperparameters, MarginalisingHyperparametersWithCosts

from sacred import Experiment



ex = Experiment('lenet_on_mnist')


@ex.config
def default_config():
    num_iterations = 20
    save_dir = "/home/kleinaa/experiments/entropy_search/benchmarks/lenet_on_mnist"
    num_restarts = 10
    Nb = 80
    Nf = 500
    Np = 100
    method = "EntropyMC"
    run_id = 0
    max_method = "CMAES"
    model_method = "GPyModel"
    environment_search = False
    mcmc = False


@ex.named_config
def entropy_config():
    method = "EntropyMC"


@ex.named_config
def hmc_config():
    model_method = "HMCGP"


@ex.named_config
def mcmc_config():
    model_method = "GPyModelMCMC"
    burnin = 200
    chain_length = 100
    n_hypers = 10
    mcmc = True


@ex.named_config
def env_entropy_config():
    method = "EnvEntropy"
    environment_search = True


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


@ex.capture
def capture_mcmc(burnin, chain_length, n_hypers):
    return burnin, chain_length, n_hypers


@ex.automain
def main(max_method, model_method, method, num_iterations, save_dir, num_restarts, mcmc, Nb, Nf, Np, run_id, environment_search):

    output_dir = os.path.join(save_dir, method + "_" + max_method + "_" + model_method, "run_" + str(run_id))

    acquisition_func = None
    task = None
    model = None
    recommendation_strategy = None
    cost_model = None
    maximizer = None

    # Define task
    if environment_search:
        task = EnvLeNetMnist()
    else:
        task = LeNetMnist()

    # Define model
    kernel = GPy.kern.Matern52(input_dim=task.n_dims)
    if model_method == "HMCGP":
        burnin, chain_length, n_hypers = capture_mcmc()
        model = HMCGP(kernel, burnin=burnin, chain_length=chain_length, n_hypers=n_hypers)
    elif model_method == "GPyModel":
        model = GPyModel(kernel, optimize=True, num_restarts=num_restarts)
    elif model_method == "GPyModelMCMC":
        burnin, chain_length, n_hypers = capture_mcmc()
        model = GPyModelMCMC(kernel, burnin=burnin, chain_length=chain_length, n_hypers=n_hypers)

    # Define cost model if we perform an environmental search
    if environment_search:
        cost_kernel = GPy.kern.Matern52(input_dim=task.n_dims)
        if model_method == "HMCGP":
            burnin, chain_length, n_hypers = capture_mcmc()
            cost_model = HMCGP(cost_kernel, burnin=burnin, chain_length=chain_length, n_hypers=n_hypers)
        elif model_method == "GPyModel":
            cost_model = GPyModel(cost_kernel, optimize=True, num_restarts=num_restarts)
        elif model_method == "GPyModelMCMC":
            burnin, chain_length, n_hypers = capture_mcmc()
            cost_model = GPyModelMCMC(cost_kernel, burnin=burnin, chain_length=chain_length, n_hypers=n_hypers)

    # Define acquisition function
    if method == "EntropyMC":
        acquisition_func = EntropyMC(model, task.X_lower, task.X_upper, optimize_posterior_mean_and_std, Nb=Nb, Nf=Nf, Np=Np)
        recommendation_strategy = optimize_posterior_mean_and_std
    elif method == "EnvEntropy":
        acquisition_func = EnvEntropySearch(model, cost_model, task.X_lower, task.X_upper, env_optimize_posterior_mean_and_std, task.is_env, Nb, Np, Nf)
    elif method == "EI":
        acquisition_func = EI(model, task.X_lower, task.X_upper, compute_incumbent)
        recommendation_strategy = compute_incumbent

    # If we perform mcmc sampling over the GPs hyperparamter we have to specify the meta acquisition function here
    if mcmc:
        if method == "EnvEntropy":
            acquisition_func = MarginalisingHyperparametersWithCosts(acquisition_func, model, cost_model)
        else:
            acquisition_func = MarginalisingHyperparameters(acquisition_func, model)

    # Define maximization function
    if max_method == "CMAES":
        maximizer = CMAES(acquisition_func, task.X_lower, task.X_upper)
    elif max_method == "Direct":
        maximizer = Direct(acquisition_func, task.X_lower, task.X_upper)
    elif max_method == "StLS":
        maximizer = StochasticLocalSearch(acquisition_func, task.X_lower, task.X_upper)

    # Define solver
    if environment_search:
        bo = EnvBayesianOptimization(acquisition_fkt=acquisition_func,
                  model=model,
                  cost_model=cost_model,
                  maximize_fkt=maximizer,
                  task=task,
                  recommendation_strategy=env_optimize_posterior_mean_and_std,
                  save_dir=output_dir,
                  num_save=1)

    else:
        bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          maximize_fkt=maximizer,
                          task=task,
                          recommendation_strategy=recommendation_strategy,
                          save_dir=output_dir,
                          num_save=1)

    # Start the experiment
    bo.run(num_iterations)

    # Compute and save test performance
    logging.info("Compute test performance...")
    _, inc = get_incumbents_over_iterations(output_dir)
    evaluate_test_performance(task, inc, os.path.join(output_dir, "test_error.npy"))
