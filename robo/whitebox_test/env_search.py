'''
Created on Jun 29, 2015

@author: Aaron Klein
'''

import GPy

import numpy as np

from robo.models.GPyModel import GPyModel
from robo.solver.env_bayesian_optimization import EnvBayesianOptimization
from robo.acquisition.EnvEntropySearch import EnvEntropySearch
from robo.maximizers.maximize import direct
from robo.recommendation.incumbent import compute_incumbent
from robo.benchmarks.synthetic_test_env_search import synthetic_fkt, get_bounds
from robo.acquisition.EntropyMC import EntropyMC
from robo.solver.bayesian_optimization import BayesianOptimization
from copy import deepcopy

import matplotlib.pyplot as plt

from robo.visualization.plotting import plot_objective_function_2d

from IPython import embed

X_lower, X_upper, n_dims, is_env_variable = get_bounds()


kernel = GPy.kern.Matern52(input_dim=n_dims)
env_es_model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)
es_model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

cost_kernel = GPy.kern.Matern52(input_dim=n_dims)
cost_model = GPyModel(cost_kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

n_representer = 10
n_hals_vals = 100
n_func_samples = 200


env_es = EnvEntropySearch(env_es_model, cost_model, X_lower=X_lower, X_upper=X_upper,
                                    is_env_variable=is_env_variable, n_representer=n_representer,
                                    n_hals_vals=n_hals_vals, n_func_samples=n_func_samples, compute_incumbent=compute_incumbent)

es = EntropyMC(es_model, X_lower, X_upper, compute_incumbent, Nb=n_representer, Nf=n_func_samples, Np=n_hals_vals)

maximizer = direct

env_bo = EnvBayesianOptimization(acquisition_fkt=env_es,
                          model=env_es_model,
                          cost_model=cost_model,
                          maximize_fkt=maximizer,
                          X_lower=X_lower,
                          X_upper=X_upper,
                          dims=n_dims,
                          objective_fkt=synthetic_fkt)

bo = BayesianOptimization(acquisition_fkt=es,
                          model=es_model,
                          maximize_fkt=maximizer,
                          X_lower=X_lower,
                          X_upper=X_upper,
                          dims=n_dims,
                          objective_fkt=synthetic_fkt)

es_X = np.array([np.random.uniform(X_lower, X_upper, n_dims)])
env_es_X = deepcopy(es_X)
es_Y = synthetic_fkt(es_X)
env_es_Y = deepcopy(es_Y)
env_bo.run(5)
# embed()
# 
# f = plt.figure()
# input
# clear
# X_lower
# clear
# resolution = 0.01
# objective_function = synthetic_fkt
# acq = np.zeros([grid1.shape[0], grid2.shape[0]])
#     grid1 = np.arange(X_lower[0], X_upper[0], resolution)
#     grid2 = np.arange(X_lower[1], X_upper[1], resolution)
# clear
# grid1.shape
# grid2.shape
# clear
# grid2
# clear
# acq = np.zeros([grid1.shape[0], grid2.shape[0]])
# acq.shape
# for i in gird1.shape[0]:
#     for j in grid2.shape[0]:
#         acq[i,j] = env_es(np.array([grid1[i], grid2[j]])
#         
#         )
# for i in grid1.shape[0]:
#     for j in grid2.shape[0]:
#         acq[i,j] = env_es(np.array([grid1[i], grid2[j]]))
# for i in range(grid1.shape[0]):
#     for j in range(grid2.shape[0]):
#         acq[i,j] = env_es(np.array([grid1[i], grid2[j]]))
# for i in range(grid1.shape[0]):
#     for j in range(grid2.shape[0]):
#         acq[i,j] = env_es(np.array([[grid1[i], grid2[j]]]))
# acq
# plt.pcolor(acq)
# acq.shape
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.pcolor(acq)
# plt.clf()
# plt.pcolor(acq)
# x = env_bo.maximize_fkt(env_es, X_lower, X_upper)
# x
# plt.plot(x[0], x[1],"ro")
# plt.plot(x[0, 0], x[0, 1],"ro")
# plt.plot(x[0, 0], x[0, 1],"ko")
# plt.plot(x[0, 0] *10 , x[0, 1]*10,"ko")
# plt.clf()
# plt.plot(x[0, 0] *10 , x[0, 1]*10,"ko")
# plt.ylim(0,100)
# plt.xlim(0,100)
# plt.plot(x[0, 0] *10 , x[0, 1]*10,"ko")
# plt.plot(x[0, 0] *100 , x[0, 1]*100,"ko")
# plt.clf()
# plt.plot(x[0, 0] *100 , x[0, 1]*100,"ko")
# plt.xlim(0,100)
# plt.ylim(0,100)
# plt.pcolor(acq)
# plt.plot(env_es.zb[:, 0]*100, env_es.zb[:, 1]*100, "bo")
# plt.plot(env_es.zb[:, 0]*100, env_es.zb[:, 1]*100, "ro")
# plt.plot(env_bo.X[:, 0] * 100, env_bo.X[:, 1] * 100, "go")
# c = np.zeros([grid1.shape[0], grid2.shape[0]])
# for i in range(grid1.shape[0]):
#     for j in range(grid2.shape[0]):
#         c[i,j] = env_es.cost_model.predict(np.array([[grid1[i], grid2[j]]]))
# c = np.zeros([grid1.shape[0], grid2.shape[0]])
# for i in range(grid1.shape[0]):
#     for j in range(grid2.shape[0]):
#         c[i,j] = env_es.cost_model.predict(np.array([[grid1[i], grid2[j]]]))[0]
# f = plt.figure()
# plt.pcolor(c)
# def ent(X):
#     new_pmin = env_es.change_pmin_by_innovation(X, env_es.f)
#             H_old = np.sum(np.multiply(self.pmin, (self.logP + self.lmb)))
#         H_new = np.sum(np.multiply(new_pmin, (np.log(new_pmin) + self.lmb)))
#         loss = np.array([[-H_new + H_old]])
# def ent(X):
#     new_pmin = env_es.change_pmin_by_innovation(X, env_es.f)
#             H_old = np.sum(np.multiply(env_es.pmin, (env_es.logP + env_es.lmb)))
#         H_new = np.sum(np.multiply(new_pmin, (np.log(new_pmin) + env_es.lmb)))
# clear
# def ent(X):
#             new_pmin = env_es.change_pmin_by_innovation(X, env_es.f)
#             H_old = np.sum(np.multiply(env_es.pmin, (env_es.logP + env_es.lmb)))
#             H_new = np.sum(np.multiply(new_pmin, (np.log(new_pmin) + env_es.lmb)))
#             loss = np.array([[-H_new + H_old]])
#     return loss
# def ent(X):
#             new_pmin = env_es.change_pmin_by_innovation(X, env_es.f)
#             H_old = np.sum(np.multiply(env_es.pmin, (env_es.logP + env_es.lmb)))
#             H_new = np.sum(np.multiply(new_pmin, (np.log(new_pmin) + env_es.lmb)))
#             loss = np.array([[-H_new + H_old]])
#             return loss
# e = np.zeros([grid1.shape[0], grid2.shape[0]])
# for i in range(grid1.shape[0]):
#     for j in range(grid2.shape[0]):
#         e[i,j] = ent(np.array([[grid1[i], grid2[j]]]))[0]
# e
# clear
# f = plt.figure()
# plt.pcolor(e)
# f = plt.figure()
# plt.bar(env_es.zb[:, 0], env_es.pmin, 0.05, color="orange")
# f = plt.figure()
# clear
# o = np.zeros([grid1.shape[0], grid2.shape[0]])
# for i in range(grid1.shape[0]):
#     for j in range(grid2.shape[0]):
#         o[i,j] = synthetic_fkt(np.array([[grid1[i], grid2[j]]]))[0]
# resolution = 0.01
# objective_function = synthetic_fkt
# paste
# grid_values
# grid_values.shape
# grid1.shape
# grid_values = np.reshape(grid_values, (grid1.shape[0], grid2.shape[0])
# )
# grid_values.shape
# clear
# X_lower
# X_upper
# plt.pcolor(grid_values)
# plt.clf()
# plt.pcolor(grid_values)
# for i in range(grid1.shape[0]):
#     for j in range(grid2.shape[0]):
#         o[i,j] = synthetic_fkt(np.array([[grid1[i], grid2[j]]]))[0]
# history
# 
# # for i in xrange(20):
# #     es_model.train(es_X, es_Y)
# #     env_es_model.train(env_es_X, env_es_Y)
# #     cost_model.train(env_es_X, np.log(env_es_X[:, 1, np.newaxis]))
# # 
# #     env_es.update(env_es_model, cost_model)
# #     #es.update(es_model)
# # 
# #     new_x = maximizer(env_es, X_lower, X_upper)
# #     new_y = synthetic_fkt(np.array(new_x))
# #     env_es_X = np.append(env_es_X, new_x, axis=0)
# #     env_es_Y = np.append(env_es_Y, new_y, axis=0)
# #     print "Env es: %s, %f"  % (new_x, new_y[0, 0])
# 
#     #new_x = maximizer(es, X_lower, X_upper)
#     #new_y = synthetic_fkt(np.array(new_x))
#     #es_X = np.append(es_X, new_x, axis=0)
#     #es_Y = np.append(es_Y, new_y, axis=0)
#     #print "Es: %s, %f" %(new_x, new_y[0, 0])
# 
# # f, (ax1) = plt.subplots(1, sharex=False)
# # ax1.plot(env_es_X[:, 0], env_es_X[:, 1],"bo")
# # #ax1.plot(es_X[:, 0], es_X[:, 1], "ro")
# # 
# # plot_objective_function_2d(synthetic_fkt, X_lower, X_upper, ax1)
# # plt.savefig("env_es_vs_es.png")