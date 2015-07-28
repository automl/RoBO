'''
Created on Jun 29, 2015

@author: Aaron Klein
'''

import GPy

import numpy as np

from robo.models.GPyModel import GPyModel
from robo.solver.env_bayesian_optimization import EnvBayesianOptimization
from robo.acquisition.EnvEntropySearch import EnvEntropySearch
from robo.maximizers.cmaes import CMAES
from robo.recommendation.incumbent import compute_incumbent
from robo.task.synthetic_test_env_search import SyntheticFktEnvSearch
from robo.task.svm_digits import EnvSVMDigits
from robo.acquisition.EntropyMC import EntropyMC
from robo.solver.bayesian_optimization import BayesianOptimization

from IPython import embed
from robo.models.GPyModelMCMC import GPyModelMCMC

#task = SyntheticFktEnvSearch()
task = EnvSVMDigits()

kernel = GPy.kern.Matern52(input_dim=task.n_dims)
env_es_model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)
#env_es_model = GPyModelMCMC(kernel, noise_variance=1e-8)
es_model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

cost_kernel = GPy.kern.Matern52(input_dim=task.n_dims)
cost_model = GPyModel(cost_kernel, optimize=True, noise_variance=1e-4, num_restarts=10)
#cost_model = GPyModelMCMC(cost_kernel, noise_variance=1e-8)
n_representer = 50
n_hals_vals = 200
n_func_samples = 500


env_es = EnvEntropySearch(env_es_model, cost_model, X_lower=task.X_lower, X_upper=task.X_upper,
                                    is_env_variable=task.is_env, n_representer=n_representer,
                                    n_hals_vals=n_hals_vals, n_func_samples=n_func_samples, compute_incumbent=compute_incumbent)

#es = EntropyMC(es_model, task.X_lower, task.X_upper, compute_incumbent, Nb=n_representer, Nf=n_func_samples, Np=n_hals_vals)

maximizer_env = CMAES(env_es, task.X_lower, task.X_upper)

env_bo = EnvBayesianOptimization(acquisition_fkt=env_es,
                          model=env_es_model,
                          cost_model=cost_model,
                          maximize_fkt=maximizer_env,
                          task=task)

#maximizer = Direct(es, task.X_lower, task.X_upper)

#bo = BayesianOptimization(acquisition_fkt=es,
#                          model=es_model,
#                          maximize_fkt=maximizer,
#                          task=task)

#es_X = np.array([np.random.uniform(task.X_lower, task.X_upper, task.n_dims)])
#env_es_X = deepcopy(es_X)
#es_Y = task.objective_function(es_X)
#env_es_Y = deepcopy(es_Y)
env_bo.run(50)
#bo.run(10)
embed()
