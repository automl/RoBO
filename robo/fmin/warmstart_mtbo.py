import os
import time
import json
import george
import logging
import numpy as np
from copy import deepcopy

from robo.models.wrapper_bohamiann import WrapperBohamiannMultiTask
from robo.models.mtbo_gp import MTBOGPMCMC
from robo.priors.env_priors import MTBOPrior
from robo.acquisition_functions.log_ei import LogEI
from robo.maximizers.differential_evolution import DifferentialEvolution
from robo.util import normalization


logger = logging.getLogger(__name__)


def transform(X, lower, upper):
    X_norm, _, _ = normalization.zero_one_normalization(X[:, :-1], lower, upper)
    X_norm = np.concatenate((X_norm, np.rint(X[:, None, -1])), axis=1)
    return X_norm


def transformation(X, acq, lower, upper):
    X_norm = transform(X, lower, upper)
    a = acq(X_norm)
    return a


def warmstart_mtbo(objective_function, lower, upper, observed_X, observed_y, n_tasks=2, num_iterations=30,
                   model_type="bohamiann", target_task_id=1, burnin=100, chain_length=200,
                   n_hypers=20, output_path=None, rng=None):
    """
    Interface to MTBO[1][2] which uses an auxiliary cheaper task to warm start the optimization on new but similar task.
    Note here we only warmstart the optimization process, in case you want to speed up Bayesian optimization by
    evaluating on auxiliary tasks during the optimization check out mtbo() or fabolas().

    [1] Multi-Task Bayesian Optimization
        K. Swersky and J. Snoek and R. Adams
        Proceedings of the 27th International Conference on Advances in Neural Information Processing Systems (NIPS'13)

    [2] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
        Bayesian Optimization with Robust Bayesian Neural Networks.
        In Advances in Neural Information Processing Systems 29 (2016).

    Parameters
    ----------
    objective_function: function
        Objective function that will be optimized
    lower: np.array(D,)
        Lower bound of the input space
    upper: np.array(D,)
        Upper bound of the input space
    observed_X: np.array(N, D + 1)
        observed point from the auxiliary task. Make sure that the last dimension identifies the auxiliary task
        (default=0). We assume the main task to have the task id = 1
    observed_y: np.array(N,)
        corresponding target values
    n_tasks: int
        Number of task
    target_task_id: int
        the id of the target task
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

    Returns
    -------
        dict with all results
    """

    assert lower.shape[0] == upper.shape[0], "Dimension miss match between upper and lower bound"

    time_start = time.time()
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    n_dims = lower.shape[0]

    # Bookkeeping
    time_func_eval = []
    time_overhead = []
    incumbents = []
    incumbent_values = []
    runtime = []

    X = deepcopy(observed_X)
    y = deepcopy(observed_y)

    if model_type == "gp_mcmc":
        # Define model for the objective function
        cov_amp = 1  # Covariance amplitude
        kernel = cov_amp

        # ARD Kernel for the configuration space
        for d in range(n_dims):
            kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                    ndim=n_dims+1, axes=d)

        task_kernel = george.kernels.TaskKernel(n_dims+1, n_dims, n_tasks)
        kernel *= task_kernel

        # Take 3 times more samples than we have hyperparameters
        if n_hypers < 2*len(kernel):
            n_hypers = 3 * len(kernel)
            if n_hypers % 2 == 1:
                n_hypers += 1

        prior = MTBOPrior(len(kernel) + 1,
                          n_ls=n_dims,
                          n_kt=len(task_kernel),
                          rng=rng)

        model_objective = MTBOGPMCMC(kernel,
                                     prior=prior,
                                     burnin_steps=burnin,
                                     chain_length=chain_length,
                                     n_hypers=n_hypers,
                                     lower=lower,
                                     upper=upper,
                                     rng=rng)
    elif model_type == "bohamiann":
        model_objective = WrapperBohamiannMultiTask(n_tasks=n_tasks)

    acquisition_func = LogEI(model_objective)

    # Optimize acquisition function only on the main task
    def wrapper(x):
        x_ = np.append(x, np.ones([x.shape[0], 1]) * target_task_id, axis=1)

        if y.shape[0] == init_points:
            eta = 0
        else:
            eta = np.min(y[init_points:])
        a = acquisition_func(x_, eta=eta)
        return a

    maximizer = DifferentialEvolution(wrapper, lower, upper)

    X = np.array(X)
    y = np.array(y)

    init_points = y.shape[0]

    for it in range(num_iterations):
        logger.info("Start iteration %d ... ", it)

        start_time = time.time()

        # Train models
        model_objective.train(X, y, do_optimize=True)

        # Maximize acquisition function
        acquisition_func.update(model_objective)

        new_x = maximizer.maximize()
        new_x = np.append(new_x, np.array([target_task_id]))

        time_overhead.append(time.time() - start_time)
        logger.info("Optimization overhead was %f seconds", time_overhead[-1])

        # Evaluate the chosen configuration
        logger.info("Evaluate candidate %s", str(new_x))
        start_time = time.time()
        new_y = objective_function(new_x[:-1], int(new_x[-1]))
        time_func_eval.append(time.time() - start_time)

        logger.info("Configuration achieved a performance of %f", new_y)
        logger.info("Evaluation of this configuration took %f seconds", time_func_eval[-1])

        # Add new observation to the data
        X = np.concatenate((X, new_x[None, :]), axis=0)
        y = np.concatenate((y, np.array([new_y])), axis=0)  # Model the target function on a logarithmic scale

        # Estimate incumbent as the best observed value so far
        best_idx = np.argmin(y[init_points:]) + init_points
        incumbent = X[best_idx][:-1]
        incumbent_value = y[best_idx]

        incumbents.append(incumbent)
        incumbent_values.append(incumbent_value)
        logger.info("Current incumbent %s with estimated performance %f", str(incumbent), incumbent_value)

        runtime.append(time.time() - time_start)

        if output_path is not None:
            data = dict()
            data["optimization_overhead"] = time_overhead[it]
            data["runtime"] = runtime[it]
            data["incumbent"] = incumbents[it].tolist()
            data["time_func_eval"] = time_func_eval[it]
            data["iteration"] = it

            json.dump(data, open(os.path.join(output_path, "mtbo_iter_%d.json" % it), "w"))

    logger.info("Final incumbent %s with estimated performance %f", str(incumbent), incumbent_value)

    results = dict()
    results["x_opt"] = incumbent.tolist()
    results["incumbents"] = [inc.tolist() for inc in incumbents]
    results["runtime"] = runtime
    results["overhead"] = time_overhead
    results["time_func_eval"] = time_func_eval
    results["incumbent_values"] = incumbent_values
    results["X"] = X
    results["y"] = y

    return results
