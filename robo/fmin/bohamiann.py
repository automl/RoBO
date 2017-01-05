import logging
import numpy as np

from robo.models.bnn import BayesianNeuralNetwork
from robo.maximizers.direct import Direct
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.pi import PI
from robo.acquisition_functions.log_ei import LogEI
from robo.acquisition_functions.lcb import LCB


logger = logging.getLogger(__name__)


def bohamiann(objective_function, lower, upper, num_iterations=30,
              acquisition_func="log_ei", n_init=3, rng=None):
    """
    General interface for Bayesian optimization for global black box optimization problems.

    Parameters
    ----------
    objective_function: function
        The objective function that is minimized. This function gets a numpy array (D,) as input and returns
        the function value (scalar)
    lower: np.ndarray (D,)
        The lower bound of the search space
    upper: np.ndarray (D,)
        The upper bound of the search space
    num_iterations: int
        The number of iterations (initial design + BO)
    acquisition_func: {"ei", "log_ei", "lcb", "pi"}
        The acquisition function
    n_init: int
        Number of points for the initial design. Make sure that it is <= num_iterations.
    rng: numpy.random.RandomState
        Random number generator

    Returns
    -------
        dict with all results
    """
    assert upper.shape[0] == lower.shape[0]
    assert n_init <= num_iterations, "Number of initial design point has to be <= than the number of iterations"

    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    model = BayesianNeuralNetwork(sampling_method="sghmc",
                                  l_rate=np.sqrt(1e-4),
                                  mdecay=0.05,
                                  burn_in=3000,
                                  n_iters=50000,
                                  precondition=True,
                                  normalize_input=True,
                                  normalize_output=True)

    if acquisition_func == "ei":
        a = EI(model)
    elif acquisition_func == "log_ei":
        a = LogEI(model)
    elif acquisition_func == "pi":
        a = PI(model)
    elif acquisition_func == "lcb":
        a = LCB(model)

    else:
        print("ERROR: %s is not a valid acquisition function!" % acquisition_func)
        return

    max_func = Direct(a, lower, upper, verbose=False)

    bo = BayesianOptimization(objective_function, lower, upper, a, model, max_func,
                              initial_points=n_init, rng=rng)

    x_best, f_min = bo.run(num_iterations)

    results = dict()
    results["x_opt"] = x_best
    results["f_opt"] = f_min
    results["incumbents"] = [inc for inc in bo.incumbents]
    results["incumbent_values"] = [val for val in bo.incumbents_values]
    results["runtime"] = bo.runtime
    results["overhead"] = bo.time_overhead
    results["X"] = bo.X
    results["y"] = bo.y
    return results
