import os
import time
import json
import logging
import numpy as np


logger = logging.getLogger(__name__)


def random_search(objective_function, lower, upper, X_init=[], Y_init=[],
                  num_iterations=30, output_path=None, rng=None):
    """
    Random Search [1] that simply evaluates random points. We do not have
    any priors thus we sample points uniformly at random.

    [1] J. Bergstra and Y. Bengio.
        Random search for hyper-parameter optimization.
        JMLR, 2012.

    Parameters
    ----------
    objective_function: function
        Objective function that will be optimized
    lower: np.array(D,)
        Lower bound of the input space
    upper: np.array(D,)
        Upper bound of the input space
    X_init: list (N, D)
            Initial points that have been already evaluated
    Y_init: list (N,1)
            Function values of the already initial points
    num_iterations: int
        Number of iterations
    output_path: string
        Specifies the path where the intermediate output after each iteration will be saved.
        If None no output will be saved to disk.
    rng: numpy.random.RandomState
        Random number generator
    Returns
    -------
    dict with all results
    """

    time_start = time.time()
    if rng is None:
        rng = np.random.RandomState()

    time_func_evals = []
    time_overhead = []
    incumbents = []
    incumbents_values = []
    runtime = []

    X = X_init
    y = Y_init

    for it in range(num_iterations):
        logger.info("Start iteration %d ... ", it)

        start_time = time.time()

        # Choose next point to evaluate
        new_x = rng.uniform(lower, upper)

        # In case of a one dim problem make x a vector
        if lower.shape[0] == 1:
            new_x = np.array([new_x])

        time_overhead.append(time.time() - start_time)

        logger.info("Optimization overhead was %f seconds", time_overhead[-1])

        # Evaluate
        logger.info("Evaluate candidate %s", str(new_x))
        start_time = time.time()
        new_y = objective_function(new_x)
        time_func_evals.append(time.time() - start_time)

        logger.info("Configuration achieved a performance of %f ", new_y)

        logger.info("Evaluation of this configuration took %f seconds", time_func_evals[-1])

        # Update the data
        X.append(new_x.tolist())
        y.append(new_y)

        # The incumbent is just the best observation we have seen so far
        best_idx = np.argmin(y)
        incumbent = X[best_idx]
        incumbent_value = y[best_idx]

        incumbents.append(incumbent)
        incumbents_values.append(incumbent_value)

        logger.info("New incumbent %s with estimated performance %f", str(incumbent), incumbent_value)

        runtime.append(time.time() - time_start)

        if output_path is not None:
            data = dict()
            data["optimization_overhead"] = time_overhead[it]
            data["runtime"] = runtime[it]
            data["incumbent"] = incumbents[it]
            data["incumbents_value"] = incumbents_values[it]
            data["time_func_eval"] = time_func_evals[it]
            data["iteration"] = it

            json.dump(data, open(os.path.join(output_path, "robo_iter_%d.json" % it), "w"))

    results = dict()
    results["x_opt"] = incumbent
    results["f_opt"] = incumbent_value
    results["incumbents"] = incumbents
    results["incumbent_values"] = incumbents_values
    results["runtime"] = runtime
    results["overhead"] = time_overhead
    results["time_func_eval"] = time_func_evals
    results["X"] = X
    results["y"] = y

    return results

