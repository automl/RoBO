'''
Created on Jul 3, 2015

@author: Aaron Klein
'''
import logging
import george
import numpy as np


from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.acquisition.information_gain_mc import InformationGainMC
from robo.acquisition.information_gain import InformationGain
from robo.acquisition.ei import EI
from robo.acquisition.lcb import LCB
from robo.acquisition.pi import PI
from robo.acquisition.log_ei import LogEI
from robo.maximizers import cmaes, direct, grid_search, stochastic_local_search
from robo.priors.default_priors import DefaultPrior
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.task.base_task import BaseTask
from robo.solver.fabolas import Fabolas
from robo.priors.env_priors import EnvPrior
from robo.acquisition.information_gain_per_unit_cost import InformationGainPerUnitCost
from robo.incumbent.best_observation import BestProjectedObservation
from robo.acquisition.integrated_acquisition import IntegratedAcquisition


logger = logging.getLogger(__name__)


def fmin(objective_func,
        X_lower,
        X_upper,
        num_iterations=30,
        maximizer="direct",
        acquisition="LogEI"):

    assert X_upper.shape[0] == X_lower.shape[0]

    class Task(BaseTask):

        def __init__(self, X_lower, X_upper, objective_fkt):
            super(Task, self).__init__(X_lower, X_upper)
            self.objective_function = objective_fkt

    task = Task(X_lower, X_upper, objective_func)

    cov_amp = 2

    initial_ls = np.ones([task.n_dims])
    exp_kernel = george.kernels.Matern52Kernel(initial_ls,
                                               ndim=task.n_dims)
    kernel = cov_amp * exp_kernel

    prior = DefaultPrior(len(kernel) + 1)

    n_hypers = 3 * len(kernel)
    if n_hypers % 2 == 1:
        n_hypers += 1
    model = GaussianProcessMCMC(kernel, prior=prior,
                                n_hypers=n_hypers,
                                chain_length=200,
                                burnin_steps=100)

    if acquisition == "EI":
        a = EI(model, X_upper=task.X_upper, X_lower=task.X_lower)
    elif acquisition == "LogEI":
        a = LogEI(model, X_upper=task.X_upper, X_lower=task.X_lower)        
    elif acquisition == "PI":
        a = PI(model, X_upper=task.X_upper, X_lower=task.X_lower)
    elif acquisition == "UCB":
        a = LCB(model, X_upper=task.X_upper, X_lower=task.X_lower)
    elif acquisition == "InformationGain":
        a = InformationGain(model, X_upper=task.X_upper, X_lower=task.X_lower)
    elif acquisition == "InformationGainMC":
        a = InformationGainMC(model, X_upper=task.X_upper, X_lower=task.X_lower,)
    else:
        logger.error("ERROR: %s is not a"
                    "valid acquisition function!" % (acquisition))
        return None
        
    acquisition_func = IntegratedAcquisition(model, a,
                                             task.X_lower,
                                             task.X_upper)        

    if maximizer == "cmaes":
        max_fkt = cmaes.CMAES(acquisition_func, task.X_lower, task.X_upper)
    elif maximizer == "direct":
        max_fkt = direct.Direct(acquisition_func, task.X_lower, task.X_upper)
    elif maximizer == "stochastic_local_search":
        max_fkt = stochastic_local_search.StochasticLocalSearch(acquisition_func,
                                                    task.X_lower,
                                                    task.X_upper)
    elif maximizer == "grid_search":
        max_fkt = grid_search.GridSearch(acquisition_func,
                                         task.X_lower,
                                         task.X_upper)
    else:
        logger.error(
            "ERROR: %s is not a valid function"
            "to maximize the acquisition function!" %
            (acquisition))
        return None

    bo = BayesianOptimization(acquisition_func=acquisition_func,
                              model=model,
                              maximize_func=max_fkt,
                              task=task)

    best_x, f_min = bo.run(num_iterations)
    return task.retransform(best_x), f_min



def fabolas_fmin(objective_func,
                 X_lower,
                 X_upper,
                 num_iterations=100,
                 n_init=40,
                 burnin=100,
                 chain_length=200,
                 Nb=50):
    """
    Interface to Fabolas [1] which models loss and training time as a
    function of dataset size and automatically trades off high information
    gain about the global optimum against computational cost.
        
    [1] Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets
        A. Klein and S. Falkner and S. Bartels and P. Hennig and F. Hutter
        http://arxiv.org/abs/1605.07079

    Parameters
    ----------
    objective_func : func
        Function handle for the objective function that get a configuration x
        and the training data subset size s and returns the validation error
        of x. See the example_fmin_fabolas.py script how the
        interface to this function should look like.
    X_lower : np.ndarray(D)
        Lower bound of the input space        
    X_upper : np.ndarray(D)
        Upper bound of the input space
    num_iterations: int
        Number of iterations for the Bayesian optimization loop
    n_init: int
        Number of points for the initial design that is run before BO starts
    burnin: int
        Determines the length of the burnin phase of the MCMC sampling
        for the GP hyperparameters
    chain_length: int
        Specifies the chain length of the MCMC sampling for the GP 
        hyperparameters
    Nb: int
        The number of representer points for approximating pmin
        
    Returns
    -------
    x : (1, D) numpy array
        The estimated global optimium also called incumbent

    """                     
                     
    assert X_upper.shape[0] == X_lower.shape[0]

    def f(x):
        x_ = x[:, :-1]
        s = x[:, -1]
        return objective_func(x_, s)

    class Task(BaseTask):

        def __init__(self, X_lower, X_upper, f):
            super(Task, self).__init__(X_lower, X_upper)
            self.objective_function = f
            is_env = np.zeros([self.n_dims])
            # Assume the last dimension to be the system size
            is_env[-1] = 1
            self.is_env = is_env

    task = Task(X_lower, X_upper, f)

    def basis_function(x):
        return (1 - x) ** 2

    # Define model for the objective function
    # Covariance amplitude
    cov_amp = 1
    
    kernel = cov_amp
    
    # ARD Kernel for the configuration space
    for d in range(task.n_dims - 1):
        kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                          ndim=task.n_dims, dim=d)

    # Kernel for the environmental variable
    # We use (1-s)**2 as basis function for the Bayesian linear kernel
    degree = 1
    env_kernel = george.kernels.BayesianLinearRegressionKernel(task.n_dims,
                                                    dim=task.n_dims - 1,
                                                    degree=degree)
    env_kernel[:] = np.ones([degree + 1]) * 0.1

    kernel *= env_kernel

    n_hypers = 3 * len(kernel)
    if n_hypers % 2 == 1:
        n_hypers += 1

    # Define the prior of the kernel's hyperparameters
    prior = EnvPrior(len(kernel) + 1,
                     n_ls=task.n_dims - 1,
                     n_lr=(degree + 1))

    model = GaussianProcessMCMC(kernel, prior=prior, burnin=burnin,
                            chain_length=chain_length,
                            n_hypers=n_hypers,
                            basis_func=basis_function,
                            dim=task.n_dims - 1)

    # Define model for the cost function
    cost_cov_amp = 3000
    
    cost_kernel = cost_cov_amp
    
    for d in range(task.n_dims - 1):
        cost_kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.1,
                                                  ndim=task.n_dims, dim=d)

    cost_degree = 1
    cost_env_kernel = george.kernels.BayesianLinearRegressionKernel(
                                                            task.n_dims,
                                                            dim=task.n_dims - 1,
                                                            degree=cost_degree)
    cost_env_kernel[:] = np.ones([cost_degree + 1]) * 0.1

    cost_kernel *= cost_env_kernel    

    cost_prior = EnvPrior(len(cost_kernel) + 1,
                          n_ls=task.n_dims - 1,
                          n_lr=(cost_degree + 1))
    cost_model = GaussianProcessMCMC(cost_kernel,
                                     prior=cost_prior,
                                     burnin=burnin,
                                     chain_length=chain_length,
                                     n_hypers=n_hypers)


    # Define acquisition function and maximizer
    es = InformationGainPerUnitCost(model, cost_model,
                              task.X_lower, task.X_upper,
                              task.is_env, Nb=Nb)

    acquisition_func = IntegratedAcquisition(model, es,
                                             task.X_lower,
                                             task.X_upper,
                                             cost_model)

    maximizer = cmaes.CMAES(acquisition_func, task.X_lower, task.X_upper)

    rec = BestProjectedObservation(model,
                                   task.X_lower,
                                   task.X_upper,
                                   task.is_env)
                                   
    bo = Fabolas(acquisition_func=acquisition_func,
                  model=model,
                  cost_model=cost_model,
                  maximize_func=maximizer,
                  task=task,
                  initial_points=n_init,
                  incumbent_estimation=rec)
    x_best = bo.run(num_iterations)
                     
    return task.retransform(x_best)