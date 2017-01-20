import logging
import george
import numpy as np

from robo.priors.default_priors import DefaultPrior
from robo.models.gaussian_process import GaussianProcess
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.maximizers.direct import Direct
from robo.maximizers.cmaes import CMAES
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.pi import PI
from robo.acquisition_functions.log_ei import LogEI
from robo.acquisition_functions.lcb import LCB
from robo.acquisition_functions.marginalization import MarginalizationGPMCMC


logger = logging.getLogger(__name__)


def bayesian_optimization(objective_function, lower, upper, num_iterations=30,
                          maximizer="direct", acquisition_func="log_ei", model_type="gp_mcmc",
                          n_init=3, rng=None):
    """
    General interface for Bayesian optimization for global black box
    optimization problems.

    Parameters
    ----------
    objective_function: function
        The objective function that is minimized. This function gets a numpy
        array (D,) as input and returns the function value (scalar)
    lower: np.ndarray (D,)
        The lower bound of the search space
    upper: np.ndarray (D,)
        The upper bound of the search space
    num_iterations: int
        The number of iterations (initial design + BO)
    maximizer: {"direct", "cmaes"}
        Defines how the acquisition function is maximized. NOTE: "cmaes" only
        works in D > 1 dimensions
    acquisition_func: {"ei", "log_ei", "lcb", "pi"}
        The acquisition function
    model_type: {"gp", "gp_mcmc", "rf"}
        The model for the objective function.
    n_init: int
        Number of points for the initial design. Make sure that it
        is <= num_iterations.
    rng: numpy.random.RandomState
        Random number generator

    Returns
    -------
        dict with all results
    """
    assert upper.shape[0] == lower.shape[0], "Dimension miss match"
    assert np.all(lower < upper), "Lower bound >= upper bound"
    assert n_init <= num_iterations, "Number of initial design point has to be <= than the number of iterations"

    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    cov_amp = 2
    n_dims = lower.shape[0]

    initial_ls = np.ones([n_dims])
    exp_kernel = george.kernels.Matern52Kernel(initial_ls,
                                               ndim=n_dims)
    kernel = cov_amp * exp_kernel

    prior = DefaultPrior(len(kernel) + 1)

    n_hypers = 3 * len(kernel)
    if n_hypers % 2 == 1:
        n_hypers += 1

    if model_type == "gp":
        model = GaussianProcess(kernel, prior=prior, rng=rng,
                                normalize_output=False, normalize_input=True,
                                lower=lower, upper=upper)
    elif model_type == "gp_mcmc":
        model = GaussianProcessMCMC(kernel, prior=prior,
                                    n_hypers=n_hypers,
                                    chain_length=200,
                                    burnin_steps=100,
                                    normalize_input=True,
                                    normalize_output=False,
                                    rng=rng, lower=lower, upper=upper)

    elif model_type == "rf":
        from robo.models.random_forest import RandomForest
        model = RandomForest(types=np.zeros([n_dims]), rng=rng)

    else:
        raise ValueError("'{}' is not a valid model".format(model))

    if acquisition_func == "ei":
        a = EI(model)
    elif acquisition_func == "log_ei":
        a = LogEI(model)
    elif acquisition_func == "pi":
        a = PI(model)
    elif acquisition_func == "lcb":
        a = LCB(model)
    else:
        raise ValueError("'{}' is not a valid acquisition function"
                         .format(acquisition_func))

    if model_type == "gp_mcmc":
        acquisition_func = MarginalizationGPMCMC(a)
    else:
        acquisition_func = a

    if maximizer == "cmaes":
        max_func = CMAES(acquisition_func, lower, upper, verbose=False,
                         rng=rng)
    elif maximizer == "direct":
        max_func = Direct(acquisition_func, lower, upper, verbose=False)
    else:
        raise ValueError("'{}' is not a valid function to maximize the "
                         "acquisition function".format(maximizer))

    bo = BayesianOptimization(objective_function, lower, upper,
                              acquisition_func, model, max_func,
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
