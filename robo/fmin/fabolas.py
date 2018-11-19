import json
import logging
import os
import time

import george
import numpy as np

from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.information_gain_per_unit_cost import InformationGainPerUnitCost
from robo.acquisition_functions.marginalization import MarginalizationGPMCMC
from robo.initial_design import init_latin_hypercube_sampling
from robo.maximizers.random_sampling import RandomSampling
from robo.models.fabolas_gp import FabolasGPMCMC
from robo.priors.env_priors import EnvPrior
from robo.util.incumbent_estimation import projected_incumbent_estimation

logger = logging.getLogger(__name__)


def transform(s, s_min, s_max):
    s_transform = (np.log2(s) - np.log2(s_min)) / (np.log2(s_max) - np.log2(s_min))
    return s_transform


def retransform(s_transform, s_min, s_max):
    s = np.rint(2 ** (s_transform * (np.log2(s_max) - np.log2(s_min)) + np.log2(s_min)))
    return int(s)


def fabolas(objective_function, lower, upper, s_min, s_max,
            n_init=40, num_iterations=100, subsets=[256, 128, 64], inc_estimation="mean",
            burnin=100, chain_length=100, n_hypers=12, output_path=None, rng=None):
    """
    Fast Bayesian Optimization of Machine Learning Hyperparameters
    on Large Datasets

    Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets
    A. Klein and S. Falkner and S. Bartels and P. Hennig and F. Hutter
    http://arxiv.org/abs/1605.07079

    Parameters
    ----------
    objective_function: function
        Objective function that will be optimized
    lower: np.array(D,)
        Lower bound of the input space
    upper: np.array(D,)
        Upper bound of the input space
    s_min: int
        Minimum number of data points for the training data set
    s_max: int
        Maximum number of data points for the training data set
    n_init: int
        Number of initial design points
    n_hypers: int
        Number of hyperparameter samples for the GP
    subsets: list
        The ratio of the subsets size of the initial design.
        For example if subsets=[256, 128, 64] then the each point of the
        initial design is evaluated on s_max/256, s_max/128 and s_max/64
    num_iterations: int
        Number of iterations
    chain_length : int
        The length of the MCMC chain for each walker.
    burnin : int
        The number of burnin steps before the actual MCMC sampling starts.
    output_path: string
        Specifies the path where the intermediate output after each iteration will be saved.
        If None no output will be saved to disk.
    rng: numpy.random.RandomState
        Random number generator
    inc_estimation: string
        Specifies how the incumbent is estimated:
            "last_seen" : determined the incumbent as the configuration that achieved the highest observed value
            "mean" :  determined the incumbent as the configuration that achieved the highest predicted mean value
    Returns
    -------
        dict
    """

    assert n_init * len(
        subsets) <= num_iterations, "Number of initial design point (n_init * len(subsets)) " \
                                    "has to be <= than the number of iterations"
    assert lower.shape[0] == upper.shape[0], "Dimension miss match between upper and lower bound"

    time_start = time.time()
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    n_dims = lower.shape[0]

    # Bookkeeping
    time_func_eval = []
    time_overhead = []
    incumbents = []
    runtime = []

    X = []
    y = []
    c = []

    # Define model for the objective function
    cov_amp = 1  # Covariance amplitude
    kernel = cov_amp

    # ARD Kernel for the configuration space
    for d in range(n_dims):
        kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                ndim=n_dims + 1, axes=d)

    # Kernel for the environmental variable
    # We use (1-s)**2 as basis function for the Bayesian linear kernel
    env_kernel = george.kernels.BayesianLinearRegressionKernel(log_a=0, log_b=0,
                                                               ndim=n_dims + 1,
                                                               axes=n_dims)
    kernel *= env_kernel

    # Take 3 times more samples than we have hyperparameters
    if n_hypers < 2 * len(kernel):
        n_hypers = 3 * len(kernel)
        if n_hypers % 2 == 1:
            n_hypers += 1

    prior = EnvPrior(len(kernel) + 1,
                     n_ls=n_dims,
                     n_lr=2,
                     rng=rng)

    quadratic_bf = lambda x: (1 - x) ** 2
    linear_bf = lambda x: x

    model_objective = FabolasGPMCMC(kernel,
                                    prior=prior,
                                    burnin_steps=burnin,
                                    chain_length=chain_length,
                                    n_hypers=n_hypers,
                                    normalize_output=False,
                                    basis_func=quadratic_bf,
                                    lower=lower,
                                    upper=upper,
                                    rng=rng)

    # Define model for the cost function
    cost_cov_amp = 1

    cost_kernel = cost_cov_amp

    # ARD Kernel for the configuration space
    for d in range(n_dims):
        cost_kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                     ndim=n_dims + 1, axes=d)

    cost_env_kernel = george.kernels.BayesianLinearRegressionKernel(log_a=0, log_b=0,
                                                                    ndim=n_dims + 1,
                                                                    axes=n_dims)
    cost_kernel *= cost_env_kernel

    cost_prior = EnvPrior(len(cost_kernel) + 1,
                          n_ls=n_dims,
                          n_lr=2,
                          rng=rng)

    model_cost = FabolasGPMCMC(cost_kernel,
                               prior=cost_prior,
                               burnin_steps=burnin,
                               chain_length=chain_length,
                               n_hypers=n_hypers,
                               basis_func=linear_bf,
                               normalize_output=False,
                               lower=lower,
                               upper=upper,
                               rng=rng)

    # Extend input space by task variable
    extend_lower = np.append(lower, 0)
    extend_upper = np.append(upper, 1)
    is_env = np.zeros(extend_lower.shape[0])
    is_env[-1] = 1

    # Define acquisition function and maximizer
    ig = InformationGainPerUnitCost(model_objective,
                                    model_cost,
                                    extend_lower,
                                    extend_upper,
                                    sampling_acquisition=EI,
                                    is_env_variable=is_env,
                                    n_representer=50)
    acquisition_func = MarginalizationGPMCMC(ig)
    maximizer = RandomSampling(acquisition_func, extend_lower, extend_upper)
    # Initial Design
    logger.info("Initial Design")
    x_init = init_latin_hypercube_sampling(lower, upper, n_init, rng)
    for it in range(n_init):

        for subset in subsets:
            start_time_overhead = time.time()
            s = int(s_max / float(subset))

            x = x_init[it]
            logger.info("Evaluate %s on subset size %d", str(x), s)
            st = time.time()
            func_val, cost = objective_function(x, s)
            time_func_eval.append(time.time() - st)

            logger.info("Configuration achieved a performance of %f with cost %f", func_val, cost)
            logger.info("Evaluation of this configuration took %f seconds", time_func_eval[-1])

            # Bookkeeping
            config = np.append(x, transform(s, s_min, s_max))
            X.append(config)
            y.append(np.log(func_val))  # Model the target function on a logarithmic scale
            c.append(np.log(cost))  # Model the cost on a logarithmic scale

            # Estimate incumbent as the best observed value so far
            best_idx = np.argmin(y)
            incumbents.append(X[best_idx][:-1])  # Incumbent is always on s=s_max

            time_overhead.append(time.time() - start_time_overhead)
            runtime.append(time.time() - time_start)

            if output_path is not None:
                data = dict()
                data["optimization_overhead"] = time_overhead[it]
                data["runtime"] = runtime[it]
                data["incumbent"] = incumbents[it].tolist()
                data["time_func_eval"] = time_func_eval[it]
                data["iteration"] = it

                json.dump(data, open(os.path.join(output_path, "fabolas_iter_%d.json" % it), "w"))

    X = np.array(X)
    y = np.array(y)
    c = np.array(c)

    for it in range(X.shape[0], num_iterations):
        logger.info("Start iteration %d ... ", it)

        start_time = time.time()

        # Train models
        model_objective.train(X, y, do_optimize=True)
        model_cost.train(X, c, do_optimize=True)

        if inc_estimation == "last_seen":
            # Estimate incumbent as the best observed value so far
            best_idx = np.argmin(y)
            incumbent = X[best_idx][:-1]
            incumbent = np.append(incumbent, 1)
            incumbent_value = y[best_idx]
        else:
            # Estimate incumbent by projecting all observed points to the task of interest and
            # pick the point with the lowest mean prediction
            incumbent, incumbent_value = projected_incumbent_estimation(model_objective, X[:, :-1],
                                                                        proj_value=1)
        incumbents.append(incumbent[:-1])
        logger.info("Current incumbent %s with estimated performance %f",
                    str(incumbent), np.exp(incumbent_value))

        # Maximize acquisition function
        acquisition_func.update(model_objective, model_cost)
        new_x = maximizer.maximize()

        s = retransform(new_x[-1], s_min, s_max)  # Map s from log space to original linear space

        time_overhead.append(time.time() - start_time)
        logger.info("Optimization overhead was %f seconds", time_overhead[-1])

        # Evaluate the chosen configuration
        logger.info("Evaluate candidate %s on subset size %f", str(new_x[:-1]), s)
        start_time = time.time()
        new_y, new_c = objective_function(new_x[:-1], s)
        time_func_eval.append(time.time() - start_time)

        logger.info("Configuration achieved a performance of %f with cost %f", new_y, new_c)
        logger.info("Evaluation of this configuration took %f seconds", time_func_eval[-1])

        # Add new observation to the data
        X = np.concatenate((X, new_x[None, :]), axis=0)
        y = np.concatenate((y, np.log(np.array([new_y]))), axis=0)  # Model the target function on a logarithmic scale
        c = np.concatenate((c, np.log(np.array([new_c]))), axis=0)  # Model the cost function on a logarithmic scale

        runtime.append(time.time() - time_start)

        if output_path is not None:
            data = dict()
            data["optimization_overhead"] = time_overhead[it]
            data["runtime"] = runtime[it]
            data["incumbent"] = incumbents[it].tolist()
            data["time_func_eval"] = time_func_eval[it]
            data["iteration"] = it

            json.dump(data, open(os.path.join(output_path, "fabolas_iter_%d.json" % it), "w"))

    # Estimate the final incumbent
    model_objective.train(X, y, do_optimize=True)
    incumbent, incumbent_value = projected_incumbent_estimation(model_objective, X[:, :-1],
                                                                proj_value=1)
    logger.info("Final incumbent %s with estimated performance %f",
                str(incumbent), incumbent_value)

    results = dict()
    results["x_opt"] = incumbent[:-1].tolist()
    results["incumbents"] = [inc.tolist() for inc in incumbents]
    results["runtime"] = runtime
    results["overhead"] = time_overhead
    results["time_func_eval"] = time_func_eval
    results["X"] = [x.tolist() for x in X]
    results["y"] = [np.exp(yi).tolist() for yi in y]
    results["c"] = [ci.tolist() for ci in c]

    return results
