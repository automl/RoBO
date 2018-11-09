import logging

import numpy as np

from robo.models.bnn import BayesianNeuralNetwork
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.lcb import LCB
from robo.acquisition_functions.log_ei import LogEI
from robo.acquisition_functions.pi import PI
from robo.maximizers.cmaes import CMAES
from robo.maximizers.differential_evolution import DifferentialEvolution
from robo.maximizers.direct import Direct
from robo.maximizers.random_sampling import RandomSampling
from robo.maximizers.scipy_optimizer import SciPyOptimizer
from robo.solver.bayesian_optimization import BayesianOptimization

logger = logging.getLogger(__name__)


def bohamiann(objective_function, lower, upper, num_iterations=30, maximizer="differential_evolution",
              acquisition_func="log_ei", n_init=3, output_path=None, rng=None):
    """
    Bohamiann uses Bayesian neural networks to model the objective function [1] inside Bayesian optimization.
    Bayesian neural networks usually scale better with the number of function evaluations and the number of dimensions
    than Gaussian processes.

    [1] Bayesian optimization with robust Bayesian neural networks
        J. T. Springenberg and A. Klein and S. Falkner and F. Hutter
        Advances in Neural Information Processing Systems 29

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
    maximizer: {"direct", "cmaes", "random", "scipy", "differential_evolution"}
        The optimizer for the acquisition function. NOTE: "cmaes" only works in D > 1 dimensions
    n_init: int
        Number of points for the initial design. Make sure that it is <= num_iterations.
    output_path: string
        Specifies the path where the intermediate output after each iteration will be saved.
        If None no output will be saved to disk.
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

    if maximizer == "cmaes":
        max_func = CMAES(a, lower, upper, verbose=True, rng=rng)
    elif maximizer == "direct":
        max_func = Direct(a, lower, upper, verbose=True)
    elif maximizer == "random":
        max_func = RandomSampling(a, lower, upper, rng=rng)
    elif maximizer == "scipy":
        max_func = SciPyOptimizer(a, lower, upper, rng=rng)
    elif maximizer == "differential_evolution":
        max_func = DifferentialEvolution(a, lower, upper, rng=rng)

    bo = BayesianOptimization(objective_function, lower, upper, a, model, max_func,
                              initial_points=n_init, output_path=output_path, rng=rng)

    x_best, f_min = bo.run(num_iterations)

    results = dict()
    results["x_opt"] = x_best
    results["f_opt"] = f_min
    results["incumbents"] = [inc for inc in bo.incumbents]
    results["incumbent_values"] = [val for val in bo.incumbents_values]
    results["runtime"] = bo.runtime
    results["overhead"] = bo.time_overhead
    results["X"] = [x.tolist() for x in bo.X]
    results["y"] = [y for y in bo.y]
    return results
