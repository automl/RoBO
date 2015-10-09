'''
Created on Jun 26, 2015

@author: Aaron Klein
'''


import GPy
from robo.models.gpy_model_mcmc import GPyModelMCMC
from robo.acquisition.ei import EI
from robo.acquisition.integrated_acquisition import IntegratedAcquisition
from robo.maximizers.direct import Direct
from robo.recommendation.incumbent import compute_incumbent
from robo.recommendation.optimize_posterior import optimize_posterior_mean_and_std
from robo.task.branin import Branin
from robo.solver.bayesian_optimization import BayesianOptimization


branin = Branin()

kernel = GPy.kern.Matern52(input_dim=branin.n_dims, ARD=True)
model = GPyModelMCMC(kernel, burnin=20, chain_length=100, n_hypers=10)

ei = EI(model, X_upper=branin.X_upper, X_lower=branin.X_lower, compute_incumbent=compute_incumbent, par=0.1)
acquisition_func = IntegratedAcquisition(model, ei)


maximizer = Direct(acquisition_func, branin.X_lower, branin.X_upper)
bo = BayesianOptimization(acquisition_func=acquisition_func,
                          model=model,
                          maximize_func=maximizer,
                          recommendation_strategy=optimize_posterior_mean_and_std,
                          task=branin)

bo.run(10)
