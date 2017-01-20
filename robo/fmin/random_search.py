import time
import logging
import numpy as np


logger = logging.getLogger(__name__)


def random_search(objective_function, lower, upper, num_iterations=30, rng=None):
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
    num_iterations: int
        Number of iterations
    rng: numpy.random.RandomState
        Random number generator
    Returns
    -------
    dict with all results
    """

    time_start = time.time()
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    time_func_eval = []
    time_overhead = []
    incumbents = []
    incumbents_values = []
    runtime = []

    X = []
    y = []

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
        time_func_eval.append(time.time() - start_time)

        logger.info("Configuration achieved a performance of %f ", new_y)

        logger.info("Evaluation of this configuration took %f seconds", time_func_eval[-1])

        # Update the data
        X.append(new_x)
        y.append(new_y)

        # The incumbent is just the best observation we have seen so far
        best_idx = np.argmin(y)
        incumbent = X[best_idx]
        incumbent_value = y[best_idx]

        incumbents.append(incumbent)
        incumbents_values.append(incumbent_value)

        logger.info("New incumbent %s with estimated performance %f", str(incumbent), incumbent_value)

        runtime.append(time.time() - time_start)

    results = dict()
    results["x_opt"] = incumbent.tolist()
    results["f_opt"] = incumbent_value
    results["incumbents"] = [inc.tolist() for inc in incumbents]
    results["runtime"] = runtime
    results["overhead"] = time_overhead
    results["time_func_eval"] = time_func_eval
    results["X"] = np.array(X)
    results["y"] = np.array(y)
    return results

