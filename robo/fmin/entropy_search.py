import logging
import george
import numpy as np

from robo.priors.default_priors import DefaultPrior
from robo.models.gaussian_process import GaussianProcess
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.maximizers.random_sampling import RandomSampling
from robo.maximizers.scipy_optimizer import SciPyOptimizer
from robo.maximizers.differential_evolution import DifferentialEvolution
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.acquisition_functions.information_gain import InformationGain
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.marginalization import MarginalizationGPMCMC
from robo.initial_design import init_latin_hypercube_sampling

logger = logging.getLogger(__name__)


def entropy_search(objective_function, lower, upper, num_iterations=30,
                   maximizer="random", model="gp_mcmc", X_init=None, Y_init=None,
                   n_init=3, output_path=None, rng=None):
    """
    Entropy search for global black box optimization problems. This is a reimplemenation of the entropy search
    algorithm by Henning and Schuler[1].

    [1] Entropy search for information-efficient global optimization.
        P. Hennig and C. Schuler.
        JMLR, (1), 2012.

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
    maximizer: {"random", "scipy", "differential_evolution"}
        Defines how the acquisition function is maximized.
    model: {"gp", "gp_mcmc"}
        The model for the objective function.
    X_init: np.ndarray(N,D)
            Initial points to warmstart BO
    Y_init: np.ndarray(N,1)
            Function values of the already initial points
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

    if model == "gp":
        gp = GaussianProcess(kernel, prior=prior, rng=rng,
                             normalize_output=False, normalize_input=True,
                             lower=lower, upper=upper)
    elif model == "gp_mcmc":
        gp = GaussianProcessMCMC(kernel, prior=prior,
                                 n_hypers=n_hypers,
                                 chain_length=200,
                                 burnin_steps=100,
                                 normalize_input=True,
                                 normalize_output=False,
                                 rng=rng, lower=lower, upper=upper)
    else:
        print("ERROR: %s is not a valid model!" % model)
        return

    a = InformationGain(gp, lower=lower, upper=upper, sampling_acquisition=EI)

    if model == "gp":
        acquisition_func = a
    elif model == "gp_mcmc":
        acquisition_func = MarginalizationGPMCMC(a)

    if maximizer == "random":
        max_func = RandomSampling(acquisition_func, lower, upper, rng=rng)
    elif maximizer == "scipy":
        max_func = SciPyOptimizer(acquisition_func, lower, upper, rng=rng)
    elif maximizer == "differential_evolution":
        max_func = DifferentialEvolution(acquisition_func, lower, upper, rng=rng)
    else:
        print("ERROR: %s is not a valid function to maximize the acquisition function!" % maximizer)
        return

    bo = BayesianOptimization(objective_function, lower, upper, acquisition_func, gp, max_func,
                              initial_design=init_latin_hypercube_sampling,
                              initial_points=n_init, rng=rng, output_path=output_path)

    x_best, f_min = bo.run(num_iterations, X=X_init, y=Y_init)

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
