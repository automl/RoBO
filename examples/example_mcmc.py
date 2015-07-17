'''
Created on Jun 26, 2015

@author: Aaron Klein
'''


import GPy
from robo.models.GPyModelMCMC import GPyModelMCMC
from robo.acquisition.EI import EI
from robo.maximizers.direct import Direct
from robo.recommendation.incumbent import compute_incumbent
from robo.task.branin import Branin
from robo.solver.bayesian_optimization import BayesianOptimization


branin = Branin()

kernel = GPy.kern.Matern52(input_dim=branin.n_dims)
model = GPyModelMCMC(kernel, burnin=20, chain_length=100, n_hypers=10)

acquisition_func = EI(model, X_upper=branin.X_upper, X_lower=branin.X_lower, compute_incumbent=compute_incumbent, par=0.1)

maximizer = Direct(acquisition_func, branin.X_lower, branin.X_upper)
bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          maximize_fkt=maximizer,
                          task=branin)

bo.run(10)
